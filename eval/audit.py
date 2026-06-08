"""
Describe 출력 품질 감사 — eval/audit.py

워크플로:
  [1] 파이프라인에서 describe=True로 실행한 결과를 JSONL trace로 저장
      (orchestrator 결과의 objects 배열을 한 줄씩 기록)
  [2] 층화 추출 + 검수용 todo 생성:
        python eval/audit.py --sample --trace path/to/trace.jsonl --n 120
  [3] todo JSON을 열어 "correct": true/false 채우기 (사람 작업)
  [4] 정확도 + 95% CI 산출:
        python eval/audit.py --score --todo path/to/audit_todo.json

trace.jsonl 한 줄 예시:
  {"image_id":"img001","object_id":0,"label":"bottle",
   "description":"a crumpled empty plastic bottle","state":"trash_signal","confidence":0.71}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from eval.stats import proportion_ci

# 층 & 오버샘플 목표 비율
STRATA_TARGETS: dict[str | None, float] = {
    "trash_signal": 0.40,   # decision-flipping → 오탐이 가장 비쌈
    "needs_wash":   0.20,
    "tidy_target":  0.15,
    "in_use":       0.15,
    None:           0.10,   # state_signal=null (기준선)
}


# ── trace 파일 로드 ────────────────────────────────────────────────────────────

def load_trace(trace_path: Path) -> list[dict]:
    """JSONL trace 로드. 줄마다 {image_id, object_id, label, description, state, ...}."""
    items = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ── 층화 추출 ─────────────────────────────────────────────────────────────────

def stratified_sample(items: list[dict], n: int = 120, seed: int = 42) -> list[dict]:
    """
    state_signal 별 층화 추출. n개 중 각 층 target 비율로 할당.
    층이 작으면 남은 할당량을 다른 층으로 재분배.
    """
    rng = random.Random(seed)

    # 층 분류
    by_stratum: dict[str | None, list[dict]] = {k: [] for k in STRATA_TARGETS}
    for item in items:
        s = item.get("state")
        stratum = s if s in STRATA_TARGETS else None
        by_stratum[stratum].append(item)

    # 1차 할당
    allocation: dict[str | None, int] = {}
    remainder = n
    for stratum, ratio in STRATA_TARGETS.items():
        tgt = min(math.ceil(n * ratio), len(by_stratum[stratum]))
        allocation[stratum] = tgt
        remainder -= tgt

    # 2차: 남은 할당량을 여유 있는 층에 추가
    if remainder > 0:
        for stratum in sorted(STRATA_TARGETS.keys(), key=lambda x: (x is None, x or "")):
            available = len(by_stratum[stratum]) - allocation[stratum]
            add = min(remainder, available)
            allocation[stratum] += add
            remainder -= add
            if remainder <= 0:
                break

    # 층별 샘플 추출
    sampled: list[dict] = []
    for stratum, k in allocation.items():
        pool = by_stratum[stratum]
        chosen = rng.sample(pool, min(k, len(pool)))
        for item in chosen:
            sampled.append({**item, "stratum": stratum})

    rng.shuffle(sampled)
    for i, s in enumerate(sampled):
        s["sample_id"] = i
        s.setdefault("correct", None)   # 사람이 채울 필드
    return sampled


# ── todo 파일 생성 ─────────────────────────────────────────────────────────────

def create_todo(trace_path: Path, output_path: Path, n: int = 120) -> None:
    items = load_trace(trace_path)
    if not items:
        print(f"[ERROR] trace 파일이 비어 있습니다: {trace_path}")
        return

    sampled = stratified_sample(items, n)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"샘플 {len(sampled)}개 저장 → {output_path}")
    strata_cnt: dict[str | None, int] = {}
    for s in sampled:
        k = s.get("stratum")
        strata_cnt[k] = strata_cnt.get(k, 0) + 1
    print("층별 분포:")
    for stratum, cnt in sorted(strata_cnt.items(), key=lambda x: -(x[1])):
        print(f"  {str(stratum):15s} {cnt:4d}개")
    print(f"\n[다음 작업] {output_path} 파일을 열어 각 항목의 'correct' 필드를 채우세요.")
    print("  correct: true   — description이 실제 상태를 올바르게 서술")
    print("  correct: false  — 틀림 또는 매우 모호")


# ── 점수 산출 ─────────────────────────────────────────────────────────────────

def score(todo_path: Path) -> None:
    with open(todo_path, encoding="utf-8") as f:
        samples = json.load(f)

    scored = [s for s in samples if s.get("correct") is not None]
    n_scored = len(scored)
    if n_scored == 0:
        print("[ERROR] 채점된 항목이 없습니다 (correct 필드가 모두 null).")
        return

    n_correct = sum(1 for s in scored if s["correct"] is True)
    lo, mid, hi = proportion_ci(n_correct, n_scored)

    print(f"\n{'='*50}")
    print(f"  describe 정확도 (전체)")
    print(f"  {n_correct}/{n_scored} = {mid:.1%}  95%CI [{lo:.1%}, {hi:.1%}]")
    print(f"{'='*50}")

    # 층별 세분화
    strata_results: dict[str | None, dict] = {}
    for s in scored:
        k = s.get("stratum")
        if k not in strata_results:
            strata_results[k] = {"n": 0, "correct": 0}
        strata_results[k]["n"] += 1
        if s["correct"]:
            strata_results[k]["correct"] += 1

    print("\n층별 정확도:")
    print(f"  {'stratum':15s}  {'correct':>10s}  {'acc':>7s}  {'95%CI':>20s}")
    for stratum, r in sorted(strata_results.items(), key=lambda x: -(x[1]["correct"]/max(x[1]["n"],1))):
        nc, ns = r["correct"], r["n"]
        l2, m2, h2 = proportion_ci(nc, ns)
        print(f"  {str(stratum):15s}  {nc:3d}/{ns:<5d}   {m2:.1%}   [{l2:.1%}, {h2:.1%}]")

    n_pending = len(samples) - n_scored
    if n_pending > 0:
        print(f"\n미채점 {n_pending}개 남음 (전체 {len(samples)}개).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Describe 출력 품질 감사",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    # sample 서브커맨드
    sp = sub.add_parser("sample", help="trace → 층화 샘플 todo 생성")
    sp.add_argument("--trace", required=True, help="trace.jsonl 경로")
    sp.add_argument("--output", default="eval/audit_todo.json",
                    help="출력 todo 파일 경로 (기본: eval/audit_todo.json)")
    sp.add_argument("--n", type=int, default=120,
                    help="목표 샘플 수 (기본: 120)")

    # score 서브커맨드
    sc = sub.add_parser("score", help="채점 완료 todo → 정확도 + CI 출력")
    sc.add_argument("--todo", required=True, help="채점된 audit_todo.json 경로")

    # 레거시 플래그 지원 (--sample / --score)
    parser.add_argument("--sample", action="store_true", help="(레거시) --trace + --n 과 함께 사용")
    parser.add_argument("--score-flag", dest="score_flag", action="store_true",
                        help="(레거시) --todo 와 함께 사용")
    parser.add_argument("--trace", help="trace.jsonl 경로")
    parser.add_argument("--todo", help="audit_todo.json 경로")
    parser.add_argument("--output", default="eval/audit_todo.json")
    parser.add_argument("--n", type=int, default=120)

    args = parser.parse_args()

    if args.cmd == "sample":
        create_todo(Path(args.trace), Path(args.output), args.n)
    elif args.cmd == "score":
        score(Path(args.todo))
    elif args.sample and args.trace:
        create_todo(Path(args.trace), Path(args.output), args.n)
    elif args.score_flag and args.todo:
        score(Path(args.todo))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
