"""
Roboflow COCO JSON → 프로젝트 GT 포맷 변환.

COCO bbox [x, y, w, h] → xyxy [x1, y1, x2, y2] 변환 + "none" 카테고리 제거.
출력: eval/ground_truth/<stem>.json (image_id + objects 구조)

실행:
  python eval/convert_roboflow_gt.py --split-dir Student-Desk-1/test
  python eval/convert_roboflow_gt.py --split-dir Student-Desk-1/test --out eval/ground_truth
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


_SKIP_LABELS = {"none", "None", ""}

# Roboflow 라벨 → OBJECT_CLASSES 라벨 매핑 (대소문자·이름 불일치 보정)
# None 값: 해당 라벨은 변환 후 제거 (매핑 불가 multi-label 등)
LABEL_MAP: dict[str, str | None] = {
    "Laptop":                        "laptop",
    "Mobilephone":                   "phone",
    "Notebook":                      "notebook",
    "Paper":                         "paper",
    "Pen":                           "pen",
    "Headphone":                     "earphones",
    "Notebook-pen-mobilephone-laptop-": None,   # multi-label → 제거
}


def convert(split_dir: Path, out_dir: Path) -> None:
    anno_path = split_dir / "_annotations.coco.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"COCO 어노테이션 파일 없음: {anno_path}")

    coco = json.loads(anno_path.read_text())

    # category_id=0 이나 이름이 "none"인 빈 카테고리 제거
    cat = {
        c["id"]: c["name"]
        for c in coco["categories"]
        if c["name"] not in _SKIP_LABELS
    }
    img = {i["id"]: i for i in coco["images"]}

    by_img: dict[int, list[dict]] = defaultdict(list)
    skipped = 0
    for a in coco["annotations"]:
        raw_label = cat.get(a["category_id"])
        if raw_label is None:
            continue
        # LABEL_MAP 적용: 매핑된 라벨 사용, None이면 제거, 없으면 소문자 그대로 시도
        if raw_label in LABEL_MAP:
            label = LABEL_MAP[raw_label]
            if label is None:
                skipped += 1
                continue
        else:
            label = raw_label.lower()
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]].append({
            "label": label,
            "bbox": [round(x), round(y), round(x + w), round(y + h)],
            "decision": None,
        })
    if skipped:
        print(f"  매핑 불가 라벨 제거: {skipped}개")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_id, objs in by_img.items():
        stem = Path(img[img_id]["file_name"]).stem
        out_path = out_dir / f"{stem}.json"
        payload = {
            "image_id": stem,
            "objects": [{"id": i, **o} for i, o in enumerate(objs)],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        written += 1

    total_objs = sum(len(v) for v in by_img.values())
    print(f"변환 완료: {written}장 → {out_dir}/  (총 {total_objs}개 객체)")
    print(f"(이미지 파일은 --images 인자로 {split_dir} 를 지정하세요)")


def main() -> None:
    p = argparse.ArgumentParser(description="Roboflow COCO GT → 프로젝트 포맷 변환")
    p.add_argument("--split-dir", required=True,
                   help="Roboflow split 디렉터리 (예: Student-Desk-1/test)")
    p.add_argument("--out", default="eval/ground_truth",
                   help="GT JSON 출력 디렉터리 (기본: eval/ground_truth)")
    args = p.parse_args()
    convert(Path(args.split_dir), Path(args.out))


if __name__ == "__main__":
    main()
