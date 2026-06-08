"""
탐지 결과로 라벨링 작업 파일(.todo.json) 자동 생성 / 완료본(.json) 변환.

[STEP 1] todo 파일 생성 (탐지기 실행 → 초안):
  python eval/make_todo.py --images eval/data/dev --output eval/ground_truth

[STEP 2] .todo.json 열어서 decision 필드 채우기 (사람 작업)
  - "decision": null  →  "discard" | "relocate" | "keep"
  - "_fp": true  로 표시하면 오탐 처리 (finalize 시 제외)
  - "missed_objects" 배열에 탐지 못한 물건 직접 추가

[STEP 3] finalize — todo → GT json 변환 + 검증:
  python eval/make_todo.py --finalize --gt eval/ground_truth

완성된 .json은 eval/compare_detectors.py --gt 로 바로 사용 가능.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from models.detector import get_detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
VALID_DECISIONS = {"discard", "relocate", "keep", "invalid"}


# ── STEP 1: todo 파일 생성 ─────────────────────────────────────────────────────

def make_todos(images_dir: Path, output_dir: Path,
               detector_name: str, conf: float) -> None:
    image_paths = sorted(p for p in images_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        print(f"[ERROR] 이미지 없음: {images_dir}"); return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{detector_name}] 탐지기 로딩...", end=" ", flush=True)
    detector = get_detector(detector_name, conf=conf)
    print("OK")

    created, skipped = 0, 0
    for img_path in image_paths:
        todo_path = output_dir / f"{img_path.stem}.todo.json"
        gt_path   = output_dir / f"{img_path.stem}.json"

        if gt_path.exists():
            print(f"  {img_path.name}: GT 완료본 있음 — 건너뜀"); skipped += 1; continue
        if todo_path.exists():
            print(f"  {img_path.name}: .todo.json 이미 있음 — 건너뜀"); skipped += 1; continue

        preds = detector.detect(str(img_path))

        # 어노테이션 이미지 저장 (라벨링 참고용)
        _save_annotated(img_path, preds, output_dir / "preview" / f"{img_path.stem}_preview.jpg")

        todo = {
            "_todo": True,
            "_labeled": False,
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "objects": [
                {
                    "id": i,
                    "label": d["label"],
                    "bbox": d["bbox"],
                    "decision": None,       # ← 사람이 채울 것
                    "_conf": d["confidence"],
                    "_fp": False,           # ← 오탐이면 true로 변경
                }
                for i, d in enumerate(preds)
            ],
            "missed_objects": [],  # ← 탐지 못한 물건 직접 추가: {"label":..., "bbox":..., "decision":...}
        }

        with open(todo_path, "w", encoding="utf-8") as f:
            json.dump(todo, f, ensure_ascii=False, indent=2)

        print(f"  {img_path.name}: {len(preds)}개 탐지 → {todo_path.name}")
        created += 1

    print(f"\n완료: {created}개 생성, {skipped}개 건너뜀")
    print(f"미리보기 이미지: {output_dir / 'preview'}")
    print("\n[다음 작업]")
    print("  1. 각 .todo.json 열어서 'decision' 필드를 채우세요")
    print("     (discard / relocate / keep 중 하나, 오탐은 _fp: true)")
    print("  2. python eval/make_todo.py --finalize --gt", output_dir)


def _save_annotated(img_path: Path, preds: list[dict], out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for d in preds:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(img, f"{d['label']} {d['_conf'] if '_conf' in d else d['confidence']:.2f}",
                    (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1)
    cv2.imwrite(str(out_path), img)


# ── STEP 3: finalize — .todo.json → .json ────────────────────────────────────

def finalize(gt_dir: Path) -> None:
    todo_paths = sorted(gt_dir.glob("*.todo.json"))
    if not todo_paths:
        print(f"[INFO] {gt_dir} 에 .todo.json 파일 없음"); return

    ok, errors = 0, []
    for todo_path in todo_paths:
        with open(todo_path, encoding="utf-8") as f:
            todo = json.load(f)

        issues = _validate_todo(todo, todo_path.name)
        if issues:
            errors.append((todo_path.name, issues))
            continue

        # _fp=true → decision="invalid" 로 변환 (GT에 보존, 지표 계산 시 제외)
        # _* 보조 키는 제거, missed_objects 병합
        gt_objects = []
        next_id = 0
        for obj in todo.get("objects", []):
            decision = "invalid" if obj.get("_fp") else obj["decision"]
            gt_objects.append({
                "id": next_id,
                "label": obj["label"],
                "bbox": obj["bbox"],
                "decision": decision,
            })
            next_id += 1
        for obj in todo.get("missed_objects", []):
            gt_objects.append({
                "id": next_id,
                "label": obj["label"],
                "bbox": obj["bbox"],
                "decision": obj["decision"],
            })
            next_id += 1

        gt = {
            "image_id": todo["image_id"],
            "objects": gt_objects,
        }

        gt_path = gt_dir / f"{todo_path.stem.removesuffix('.todo')}.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        todo_path.unlink()  # todo 파일 삭제
        print(f"  ✓  {todo_path.name} → {gt_path.name}  ({len(gt_objects)}개 객체)")
        ok += 1

    print(f"\n변환 완료: {ok}개")
    if errors:
        print(f"\n[오류] {len(errors)}개 파일 — 수정 후 다시 실행:")
        for fname, issues in errors:
            print(f"  {fname}:")
            for issue in issues:
                print(f"    · {issue}")


def _validate_todo(todo: dict, fname: str) -> list[str]:
    issues = []
    for obj in todo.get("objects", []):
        if obj.get("_fp"):
            continue
        if obj.get("decision") not in VALID_DECISIONS:
            issues.append(
                f"id={obj.get('id')} label={obj.get('label')!r}: "
                f"decision={obj.get('decision')!r} (discard/relocate/keep 중 하나여야 함)"
            )
    for obj in todo.get("missed_objects", []):
        if obj.get("decision") not in VALID_DECISIONS:
            issues.append(
                f"missed_objects label={obj.get('label')!r}: "
                f"decision={obj.get('decision')!r} 누락"
            )
    return issues


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="라벨링 .todo.json 생성 / finalize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--images", default=None,
                        help="[make] 이미지 디렉터리 (예: eval/data/dev)")
    parser.add_argument("--output", default="eval/ground_truth",
                        help="[make] .todo.json 저장 경로 (기본: eval/ground_truth)")
    parser.add_argument("--detector", default="yolo-world",
                        choices=["yolo-world", "yoloe"],
                        help="[make] 탐지기 (기본: yolo-world)")
    parser.add_argument("--conf", type=float, default=0.08,
                        help="[make] confidence threshold (기본: 0.08)")
    parser.add_argument("--finalize", action="store_true",
                        help="[finalize] .todo.json → .json 변환")
    parser.add_argument("--gt", default="eval/ground_truth",
                        help="[finalize] .todo.json 디렉터리 (기본: eval/ground_truth)")
    args = parser.parse_args()

    if args.finalize:
        finalize(Path(args.gt))
    elif args.images:
        make_todos(
            images_dir=Path(args.images),
            output_dir=Path(args.output),
            detector_name=args.detector,
            conf=args.conf,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
