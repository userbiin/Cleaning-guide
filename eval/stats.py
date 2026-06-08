"""
통계 함수 모음 — eval/stats.py

사용:
  from eval.stats import compute_kappa, proportion_ci

  kappa = compute_kappa(labels_a, labels_b)
  lo, mid, hi = proportion_ci(n_correct, n_total)

CLI (파일 두 개 받아 kappa 산출):
  python eval/stats.py kappa --file-a a.txt --file-b b.txt

  python eval/stats.py ci --n-correct 98 --n-total 120
"""

from __future__ import annotations

import math
import sys


# ── Wilson score 비율 신뢰구간 ─────────────────────────────────────────────────

def proportion_ci(
    n: int,
    N: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Wilson score 신뢰구간.
    Returns (lower_bound, point_estimate, upper_bound).
    소표본(N<30)에서도 안정적.
    """
    if N == 0:
        return (0.0, 0.0, 0.0)

    try:
        from scipy.stats import norm as sp_norm
        z = float(sp_norm.ppf((1 + confidence) / 2))
    except ImportError:
        z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)

    p = n / N
    z2 = z * z
    denom  = 1 + z2 / N
    center = (p + z2 / (2 * N)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / N + z2 / (4 * N * N))
    return (max(0.0, center - margin), p, min(1.0, center + margin))


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def compute_kappa(y1: list, y2: list) -> float:
    """
    Cohen's kappa 2인 라벨러 일치도.
    목표: κ ≥ 0.7 (Substantial). κ < 0.6이면 가이드 보완 필요.
    """
    if len(y1) != len(y2):
        raise ValueError(f"y1, y2 길이 불일치: {len(y1)} vs {len(y2)}")
    if len(y1) == 0:
        return float("nan")

    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y1, y2))
    except ImportError:
        pass

    # sklearn 미설치 시 직접 계산
    from collections import Counter
    n = len(y1)
    p_o = sum(a == b for a, b in zip(y1, y2)) / n
    labels = sorted(set(y1) | set(y2))
    c1, c2 = Counter(y1), Counter(y2)
    p_e = sum(c1[lbl] / n * c2[lbl] / n for lbl in labels)
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0


def kappa_interpretation(kappa: float) -> str:
    """Landis & Koch (1977) kappa 해석 범주."""
    if math.isnan(kappa): return "N/A"
    if kappa < 0:         return "Poor (0 미만)"
    if kappa < 0.20:      return "Slight"
    if kappa < 0.40:      return "Fair"
    if kappa < 0.60:      return "Moderate"
    if kappa < 0.80:      return "Substantial"
    return "Almost Perfect"


# ── 정규 근사 비율 차이 검정 (A/B describe 실험 보조) ────────────────────────

def proportion_ztest(n1: int, N1: int, n2: int, N2: int) -> tuple[float, float]:
    """
    두 비율 p1=n1/N1, p2=n2/N2의 차이에 대한 정규 근사 z-검정.
    Returns (z_stat, two_sided_p_value).
    """
    p1, p2 = n1 / N1, n2 / N2
    p_pool = (n1 + n2) / (N1 + N2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/N1 + 1/N2))
    if se == 0:
        return (float("nan"), float("nan"))
    z = (p1 - p2) / se
    try:
        from scipy.stats import norm as sp_norm
        p_val = float(2 * (1 - sp_norm.cdf(abs(z))))
    except ImportError:
        # 근사 (z-table)
        p_val = float("nan")
    return (round(z, 4), round(p_val, 4))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="통계 유틸")
    sub = parser.add_subparsers(dest="cmd")

    # kappa 서브커맨드
    kp = sub.add_parser("kappa", help="두 라벨 파일의 Cohen's kappa 계산")
    kp.add_argument("--file-a", required=True, help="라벨러 A 파일 (줄마다 레이블)")
    kp.add_argument("--file-b", required=True, help="라벨러 B 파일 (줄마다 레이블)")

    # ci 서브커맨드
    ci_p = sub.add_parser("ci", help="비율 Wilson 95% 신뢰구간 계산")
    ci_p.add_argument("--n-correct", type=int, required=True, help="정확한 수")
    ci_p.add_argument("--n-total",   type=int, required=True, help="전체 수")
    ci_p.add_argument("--confidence", type=float, default=0.95, help="신뢰수준 (기본 0.95)")

    args = parser.parse_args()

    if args.cmd == "kappa":
        y1 = Path(args.file_a).read_text(encoding="utf-8").split()
        y2 = Path(args.file_b).read_text(encoding="utf-8").split()
        k = compute_kappa(y1, y2)
        print(f"Cohen's kappa = {k:.4f}  [{kappa_interpretation(k)}]")
        print(f"목표: κ ≥ 0.70 (Substantial)")

    elif args.cmd == "ci":
        lo, mid, hi = proportion_ci(args.n_correct, args.n_total, args.confidence)
        pct = args.confidence * 100
        print(f"비율  = {mid:.4f}  ({args.n_correct}/{args.n_total})")
        print(f"{pct:.0f}% CI = [{lo:.4f}, {hi:.4f}]")
    else:
        parser.print_help()


if __name__ == "__main__":
    from pathlib import Path
    main()
