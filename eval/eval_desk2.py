"""
Working Desk 평가 v2 — polygon 파싱 수정 + Location Recall + Confusion Matrix

실행:
  python eval/eval_desk2.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

CONSOLIDATED_CLASSES = [
    "book", "pen", "pencil", "marker", "scissors", "ruler", "eraser",
    "laptop", "keyboard", "mouse", "monitor", "phone",
    "tablet", "cable", "earphones", "remote control",
    "cup", "bottle", "plate", "bowl", "spoon", "fork", "snack",
    "clothes", "socks", "bag", "backpack", "hat", "shoe",
    "cosmetics", "tissue", "toothbrush", "comb", "mirror",
    "trash", "box", "toy",
]

DESK_TO_OURS: dict[str, str] = {
    "bottle": "bottle", "cell phone": "phone", "cup": "cup",
    "cutter": "scissors", "fork": "fork", "keyboard": "keyboard",
    "knife": "scissors", "mouse": "mouse", "remote": "remote control",
    "scissors": "scissors", "spoon": "spoon",
}

IOU_THR = 0.5
DS_DIR   = _ROOT / "eval" / "data" / "desk"
OUT_DIR  = _ROOT / "outputs" / "desk_eval"


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _bar(v: float, w: int = 30) -> str:
    return "█" * int(v * w)


def _parse_gt(lbl_path: Path, W: int, H: int, class_names: list) -> list[dict]:
    """polygon/bbox 자동 판별 후 xyxy 픽셀 bbox 반환."""
    objects = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_idx = int(parts[0])
        coords  = list(map(float, parts[1:]))

        if len(coords) > 4:          # polygon: x1 y1 x2 y2 ... xn yn
            xs = coords[0::2]
            ys = coords[1::2]
            x1, y1 = int(min(xs) * W), int(min(ys) * H)
            x2, y2 = int(max(xs) * W), int(max(ys) * H)
        else:                        # bbox: cx cy w h
            cx, cy, bw, bh = coords
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)

        raw   = class_names[cls_idx] if cls_idx < len(class_names) else ""
        ours  = DESK_TO_OURS.get(raw)
        if ours is None:
            continue
        objects.append({"label_desk": raw, "label_ours": ours,
                        "bbox": [x1, y1, x2, y2]})
    return objects


# ── 추론 ─────────────────────────────────────────────────────────────────────

def run(img_dir: Path, lbl_dir: Path, class_names: list) -> list[dict]:
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cpu"
    )
    print(f"[추론] device={device}  conf=0.08  imgsz=1280  iou=0.7  agnostic_nms=True")

    model = YOLO("yolov8s-world.pt")
    model.set_classes(CONSOLIDATED_CLASSES + [""])

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    n = len(img_paths)
    all_results = []

    for i, img_path in enumerate(img_paths):
        from PIL import Image
        try:
            img = Image.open(img_path)
            W, H = img.size
        except Exception:
            W, H = 640, 640

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_objects = _parse_gt(lbl_path, W, H, class_names) if lbl_path.exists() else []

        try:
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
                    preds.append({"label": lbl,
                                  "confidence": round(float(box.conf[0]), 3),
                                  "bbox": list(map(int, box.xyxy[0]))})
        except Exception as e:
            preds = []
            print(f"  [오류] {img_path.name}: {e}")

        # 클래스 일치 매칭
        matched_gt = [False] * len(gt_objects)
        for gi, gt in enumerate(gt_objects):
            for pred in preds:
                if pred["label"] == gt["label_ours"] and \
                        _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    matched_gt[gi] = True
                    break

        all_results.append({"image_path": str(img_path),
                             "gt": gt_objects, "preds": preds,
                             "matched_gt": matched_gt})

        if (i + 1) % 50 == 0 or (i + 1) == n:
            hit = sum(sum(r["matched_gt"]) for r in all_results)
            tot = sum(len(r["gt"]) for r in all_results)
            print(f"  [{i+1:>4}/{n}] 누적 Recall {hit/tot*100:.1f}%  ({hit}/{tot})")

    return all_results


# ── 분석 ─────────────────────────────────────────────────────────────────────

def analyze(all_results: list[dict]) -> None:
    from collections import defaultdict

    # ── 1. 클래스 Recall (수정된 파싱) ───────────────────────────────────────
    cls_gt:  dict[str, int] = defaultdict(int)
    cls_hit: dict[str, int] = defaultdict(int)
    for r in all_results:
        for gi, gt in enumerate(r["gt"]):
            cls_gt[gt["label_ours"]] += 1
            if r["matched_gt"][gi]:
                cls_hit[gt["label_ours"]] += 1

    total_gt  = sum(cls_gt.values())
    total_hit = sum(cls_hit.values())
    overall   = total_hit / total_gt if total_gt else 0.0
    cls_recall_asc = sorted(
        {c: cls_hit[c]/cls_gt[c] for c in cls_gt}.items(), key=lambda x: x[1]
    )

    # ── 2. Location Recall (클래스 무관 IoU≥0.5) ────────────────────────────
    loc_hit = 0
    loc_total = 0
    for r in all_results:
        for gt in r["gt"]:
            loc_total += 1
            for pred in r["preds"]:
                if _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    loc_hit += 1
                    break
    loc_recall = loc_hit / loc_total if loc_total else 0.0

    # ── 3. Confusion Matrix (IoU≥0.5 매칭 쌍) ───────────────────────────────
    # gt_label → { pred_label: count }
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        for gt in r["gt"]:
            best_iou, best_pred_lbl = 0.0, None
            for pred in r["preds"]:
                iou_val = _iou(gt["bbox"], pred["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred_lbl = pred["label"]
            if best_iou >= IOU_THR and best_pred_lbl:
                confusion[gt["label_ours"]][best_pred_lbl] += 1
            elif best_iou < IOU_THR:
                confusion[gt["label_ours"]]["[위치 미탐지]"] += 1

    # ── 출력 ────────────────────────────────────────────────────────────────
    SEP = "=" * 62
    lines = [
        SEP,
        "  Working Desk 평가 v2 (polygon 파싱 수정)",
        SEP,
        f"  평가 이미지:    {len(all_results)}장",
        f"  GT 객체 총수:   {total_gt}개",
        "",
        f"[클래스 Recall — 수정 후]",
        f"  전체 Recall: {overall*100:.1f}%  (수정 전: 0.4%)",
        "",
    ]
    for cls, rc in cls_recall_asc:
        lines.append(f"  {cls:<22} {rc*100:5.1f}% |{_bar(rc)}")

    lines += [
        "",
        f"[Location Recall (클래스 무관, IoU≥0.5)]",
        f"  GT {loc_total}개 중 위치 탐지 성공: {loc_hit}개",
        f"  Location Recall: {loc_recall*100:.1f}%",
        f"  → 클래스 Recall({overall*100:.1f}%)과의 차이: "
        f"{(loc_recall - overall)*100:.1f}%p",
        f"     (차이 = 위치는 맞는데 클래스명이 틀린 케이스)",
        "",
        "[Confusion Matrix (IoU≥0.5 매칭 기준)]",
        f"  GT 클래스         예측 클래스 분포",
        f"  {'-'*54}",
    ]
    for gt_cls in sorted(confusion.keys()):
        total_c = sum(confusion[gt_cls].values())
        top = sorted(confusion[gt_cls].items(), key=lambda x: -x[1])[:5]
        dist = "  |  ".join(f"{p}×{n}" for p, n in top)
        lines.append(f"  {gt_cls:<20}  (총 {total_c:>3}개) → {dist}")

    lines.append(SEP)
    summary = "\n".join(lines)
    print("\n" + summary)

    # ── 파일 저장 ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # location_recall.txt
    loc_lines = [
        "Location Recall (클래스 무관, IoU≥0.5)",
        "=" * 40,
        f"GT 총수:          {loc_total}개",
        f"위치 탐지 성공:   {loc_hit}개",
        f"Location Recall:  {loc_recall*100:.1f}%",
        f"클래스 Recall:    {overall*100:.1f}%",
        f"갭 (위치O 클래스X): {(loc_recall-overall)*100:.1f}%p",
        "",
        "클래스별 Location Recall (IoU≥0.5, 클래스 무관):",
    ]
    loc_cls: dict[str, int] = defaultdict(int)
    loc_cls_hit: dict[str, int] = defaultdict(int)
    for r in all_results:
        for gt in r["gt"]:
            loc_cls[gt["label_ours"]] += 1
            for pred in r["preds"]:
                if _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    loc_cls_hit[gt["label_ours"]] += 1
                    break
    for cls in sorted(loc_cls.keys()):
        lrc = loc_cls_hit[cls] / loc_cls[cls]
        crc = cls_hit[cls] / cls_gt[cls]
        loc_lines.append(
            f"  {cls:<22} locR={lrc*100:.1f}%  clsR={crc*100:.1f}%  "
            f"gap={( lrc-crc)*100:.1f}%p"
        )
    (OUT_DIR / "location_recall.txt").write_text(
        "\n".join(loc_lines), encoding="utf-8"
    )

    # confusion_matrix.csv
    all_pred_labels = sorted({p for d in confusion.values() for p in d})
    with open(OUT_DIR / "confusion_matrix.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["GT \\ PRED"] + all_pred_labels)
        for gt_cls in sorted(confusion.keys()):
            row = [gt_cls] + [confusion[gt_cls].get(p, 0) for p in all_pred_labels]
            writer.writerow(row)

    print(f"\n[저장 완료]")
    print(f"  {OUT_DIR}/location_recall.txt")
    print(f"  {OUT_DIR}/confusion_matrix.csv")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    yaml_path = DS_DIR / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    class_names: list[str] = meta.get("names", [])

    # test 우선, 없으면 valid
    for split in ("test", "valid"):
        img_dir = DS_DIR / split / "images"
        lbl_dir = DS_DIR / split / "labels"
        if img_dir.exists():
            break

    n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"split='{split}'  이미지 {n_imgs}장  클래스: {class_names}")

    all_results = run(img_dir, lbl_dir, class_names)
    analyze(all_results)


if __name__ == "__main__":
    main()
