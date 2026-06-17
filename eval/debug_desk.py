"""
Working Desk GT annotation 포맷 확인 및 bbox 매칭 디버깅 — eval/debug_desk.py

실행:
  python eval/debug_desk.py
"""

import json
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.detector import OBJECT_CLASSES

CONSOLIDATED_CLASSES = [
    "book", "pen", "pencil", "marker", "scissors", "ruler", "eraser",
    "laptop", "keyboard", "mouse", "monitor", "phone",
    "tablet", "cable", "earphones", "remote control",
    "cup", "bottle", "plate", "bowl", "spoon", "fork", "snack",
    "clothes", "socks", "bag", "backpack", "hat", "shoe",
    "cosmetics", "tissue", "toothbrush", "comb", "mirror",
    "trash", "box", "toy",
]

DESK_TO_OURS = {
    "bottle": "bottle", "cell phone": "phone", "cup": "cup",
    "cutter": "scissors", "fork": "fork", "keyboard": "keyboard",
    "knife": "scissors", "mouse": "mouse", "remote": "remote control",
    "scissors": "scissors", "spoon": "spoon",
}

IOU_THR = 0.5
OUT_DIR = _ROOT / "outputs" / "desk_eval" / "debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DS_DIR = _ROOT / "eval" / "data" / "desk"


def _iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def parse_label_polygon(line: str, W: int, H: int, class_names: list) -> dict | None:
    """polygon 포맷: class_idx x1 y1 x2 y2 ... xn yn (정규화)"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_idx = int(parts[0])
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x1, y1 = int(min(xs) * W), int(min(ys) * H)
    x2, y2 = int(max(xs) * W), int(max(ys) * H)
    raw_label = class_names[cls_idx] if cls_idx < len(class_names) else ""
    our_label = DESK_TO_OURS.get(raw_label)
    return {
        "label_desk": raw_label,
        "label_ours": our_label,
        "bbox": [x1, y1, x2, y2],
    }


def parse_label_bbox(line: str, W: int, H: int, class_names: list) -> dict | None:
    """표준 YOLOv8 bbox 포맷: class_idx cx cy w h (정규화)"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_idx = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    raw_label = class_names[cls_idx] if cls_idx < len(class_names) else ""
    our_label = DESK_TO_OURS.get(raw_label)
    return {
        "label_desk": raw_label,
        "label_ours": our_label,
        "bbox": [x1, y1, x2, y2],
    }


def detect_label_format(lbl_path: Path) -> str:
    """GT 포맷 자동 판별: polygon vs bbox"""
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) > 5:
            return "polygon"
        if len(parts) == 5:
            return "bbox"
    return "bbox"


def draw_debug_image(img_path: Path, gt_objects: list, preds: list, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # GT: 초록
    for gt in gt_objects:
        b = gt["bbox"]
        draw.rectangle(b, outline="lime", width=3)
        draw.text((b[0]+2, b[1]+2), f"GT:{gt['label_desk']}", fill="lime", font=font)

    # PRED: 빨강
    for pred in preds[:8]:
        b = pred["bbox"]
        draw.rectangle(b, outline="red", width=2)
        label = f"{pred['label']}({pred['confidence']:.2f})"
        draw.text((b[0]+2, b[3]-18), label, fill="red", font=font)

    img.save(out_path)
    print(f"  저장: {out_path.name}")


def main():
    yaml_path = DS_DIR / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    class_names: list[str] = meta.get("names", [])

    img_dir = DS_DIR / "test" / "images"
    lbl_dir = DS_DIR / "test" / "labels"
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    # ── 1. GT 포맷 확인 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("[1] GT annotation 포맷 확인")
    print("=" * 60)

    sample_lbl = None
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        if lbl_file.read_text().strip():
            sample_lbl = lbl_file
            break

    if sample_lbl:
        lines = sample_lbl.read_text().strip().splitlines()
        fmt = detect_label_format(sample_lbl)
        print(f"  파일: {sample_lbl.name}")
        print(f"  첫 줄 (공백 구분 값 수): {len(lines[0].split())}개")
        print(f"  → 포맷 판별: {fmt.upper()}")
        print(f"  원본 첫 줄: {lines[0][:80]}...")
        if fmt == "polygon":
            print(f"  해석: class_idx x1 y1 x2 y2 ... xn yn  (polygon, 꼭짓점 나열)")
            print(f"  → 현재 eval_desk.py는 bbox 포맷으로 읽어서 bbox가 틀림!")
        else:
            print(f"  해석: class_idx cx cy w h  (표준 YOLOv8 bbox)")
    print()

    # ── 2. 샘플 이미지 디버깅 ────────────────────────────────────────────────
    print("=" * 60)
    print("[2] 샘플 이미지 bbox 디버깅")
    print("=" * 60)

    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s-world.pt")
    model.set_classes(CONSOLIDATED_CLASSES + [""])

    # GT가 있는 샘플 3장 선택
    debugged = 0
    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists() or not lbl_path.read_text().strip():
            continue

        img = Image.open(img_path)
        W, H = img.size
        fmt = detect_label_format(lbl_path)
        parser = parse_label_polygon if fmt == "polygon" else parse_label_bbox

        gt_objects = []
        for line in lbl_path.read_text().strip().splitlines():
            obj = parser(line, W, H, class_names)
            if obj and obj["label_ours"]:
                gt_objects.append(obj)

        if not gt_objects:
            continue

        # 추론
        results = model.predict(
            source=str(img_path), conf=0.08, imgsz=1280,
            iou=0.7, agnostic_nms=True, verbose=False, device=device,
        )
        raw = results[0]
        preds = []
        if raw.boxes is not None:
            for box in raw.boxes:
                lbl = raw.names[int(box.cls[0])]
                if lbl == "":
                    continue
                preds.append({
                    "label": lbl,
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": list(map(int, box.xyxy[0])),
                })

        print(f"\n  이미지: {img_path.name}  ({W}x{H})  포맷: {fmt}")
        print(f"  (a) GT ({len(gt_objects)}개):")
        for gt in gt_objects:
            print(f"      {gt['label_desk']:15} → {gt['label_ours']:15} bbox={gt['bbox']}")

        print(f"  (b) PRED ({len(preds)}개, 상위 5):")
        for p in preds[:5]:
            print(f"      {p['label']:20} conf={p['confidence']:.3f}  bbox={p['bbox']}")

        print(f"  (c) IoU 매칭:")
        for gt in gt_objects:
            best_iou, best_pred = 0.0, None
            for p in preds:
                iou_val = _iou(gt["bbox"], p["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred = p
            matched = best_iou >= IOU_THR and best_pred and best_pred["label"] == gt["label_ours"]
            status = "✓ HIT" if matched else "✗ MISS"
            pred_info = f"{best_pred['label']}(IoU={best_iou:.3f})" if best_pred else "없음"
            print(f"      GT {gt['label_ours']:15} ↔ best pred: {pred_info}  {status}")

        # (d) 시각화
        out_path = OUT_DIR / f"debug_{img_path.stem[:30]}.jpg"
        draw_debug_image(img_path, gt_objects, preds, out_path)

        debugged += 1
        if debugged >= 3:
            break

    # ── 3. 요약 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[3] 결론 및 수정 필요 사항")
    print(f"{'='*60}")
    fmt_final = detect_label_format(sample_lbl) if sample_lbl else "unknown"
    if fmt_final == "polygon":
        print("  ✗ eval_desk.py가 bbox 포맷으로 읽고 있으나 실제는 polygon")
        print("  → parse_label_polygon() 방식으로 수정하면 정상 Recall 예상")
    else:
        print("  ✓ bbox 포맷 정상 — 매칭 실패는 다른 원인")
        print("  → 위 IoU 결과에서 best_iou 값 확인 필요")
    print(f"\n  디버그 이미지 저장 위치: {OUT_DIR}/")


if __name__ == "__main__":
    main()
