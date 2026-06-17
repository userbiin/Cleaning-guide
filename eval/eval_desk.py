"""
Detecting Working Desk (Roboflow) 기반 YOLO-World Recall 평가 — eval/eval_desk.py

실행:
  python eval/eval_desk.py --api-key i6JLRbdc9uMSjXk4anbw
  python eval/eval_desk.py --api-key i6JLRbdc9uMSjXk4anbw --skip-download  # 재실행 시
"""

import argparse
import csv
import json
import sys
import yaml
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# ── 통합 OBJECT_CLASSES (COCO 평가와 동일) ───────────────────────────────────
CONSOLIDATED_CLASSES = [
    "book", "pen", "pencil", "marker", "scissors", "ruler", "eraser",
    "laptop", "keyboard", "mouse", "monitor", "phone",
    "tablet", "cable", "earphones", "remote control",
    "cup", "bottle", "plate", "bowl", "spoon", "fork", "snack",
    "clothes", "socks", "bag", "backpack", "hat", "shoe",
    "cosmetics", "tissue", "toothbrush", "comb", "mirror",
    "trash", "box", "toy",
]

# ── Working Desk GT 클래스 → 통합 클래스 매핑 ────────────────────────────────
DESK_TO_OURS: dict[str, str] = {
    "bottle":     "bottle",
    "cell phone": "phone",
    "cup":        "cup",
    "cutter":     "scissors",
    "fork":       "fork",
    "keyboard":   "keyboard",
    "knife":      "scissors",
    "mouse":      "mouse",
    "remote":     "remote control",
    "scissors":   "scissors",
    "spoon":      "spoon",
}

LVIS_RECALL = 25.2  # 이전 LVIS 평가 결과
IOU_THR = 0.5


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _bar(value: float, width: int = 30) -> str:
    return "█" * int(value * width)


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                  W: int, H: int) -> list[int]:
    """YOLOv8 정규화 xywh → 픽셀 xyxy"""
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    return [max(0, x1), max(0, y1), min(W, x2), min(H, y2)]


# ── 1단계: Roboflow 다운로드 ──────────────────────────────────────────────────

def download_dataset(api_key: str) -> Path:
    from roboflow import Roboflow

    print("Roboflow에서 Detecting Working Desk 다운로드 중...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("s-workspace-ylfnl").project("Detecting working desk")
    version = project.version(1)
    ds = version.download("yolov8", location=str(_ROOT / "eval" / "data" / "desk"))

    ds_dir = Path(ds.location)
    yaml_path = ds_dir / "data.yaml"

    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    class_names: list[str] = meta.get("names", [])
    print(f"\n[1단계 완료]")
    print(f"  다운로드 경로: {ds_dir}")
    print(f"  클래스 ({len(class_names)}개): {class_names}")

    # split 확인
    for split in ("test", "valid", "train"):
        split_dir = ds_dir / split / "images"
        if split_dir.exists():
            imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            print(f"  split 사용: '{split}'  이미지 {len(imgs)}장")
            break

    # 전체 디렉터리 구조
    print(f"\n  디렉터리 구조:")
    for p in sorted(ds_dir.rglob("*")):
        if p.is_dir():
            print(f"    {p.relative_to(ds_dir)}/")

    return ds_dir


def find_eval_split(ds_dir: Path) -> tuple[Path, Path, list[str]]:
    """평가할 split의 images/labels 디렉터리와 클래스 목록 반환"""
    yaml_path = ds_dir / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    class_names: list[str] = meta.get("names", [])

    for split in ("test", "valid", "train"):
        img_dir = ds_dir / split / "images"
        lbl_dir = ds_dir / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            return img_dir, lbl_dir, class_names

    raise RuntimeError(f"test/valid/train 디렉터리를 찾을 수 없음: {ds_dir}")


# ── 2단계: 추론 ───────────────────────────────────────────────────────────────

def run_inference(ds_dir: Path) -> list[dict]:
    from ultralytics import YOLO
    from PIL import Image
    import torch

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cpu"
    )

    img_dir, lbl_dir, class_names = find_eval_split(ds_dir)
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    n = len(img_paths)

    print(f"\n[2단계] YOLO-World 로딩... device={device}")
    model = YOLO("yolov8s-world.pt")
    model.set_classes(CONSOLIDATED_CLASSES + [""])
    print(f"  conf=0.08  imgsz=1280  iou=0.7  agnostic_nms=True")
    print(f"  평가 이미지: {n}장  |  클래스: {class_names}")

    all_results = []

    for i, img_path in enumerate(img_paths):
        from PIL import Image as PILImage
        try:
            img = PILImage.open(img_path)
            W, H = img.size
        except Exception:
            W, H = 640, 640

        # GT 파싱 (YOLOv8 label: class_idx cx cy w h)
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_objects = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_idx = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                raw_label = class_names[cls_idx] if cls_idx < len(class_names) else ""
                our_label = DESK_TO_OURS.get(raw_label)
                if our_label is None:
                    continue
                gt_objects.append({
                    "label_desk": raw_label,
                    "label_ours": our_label,
                    "bbox": _yolo_to_xyxy(cx, cy, bw, bh, W, H),
                })

        # 추론
        try:
            results = model.predict(
                source=str(img_path),
                conf=0.08,
                imgsz=1280,
                iou=0.7,
                agnostic_nms=True,
                verbose=False,
                device=device,
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
        except Exception as e:
            preds = []
            print(f"  [추론 오류] {img_path.name}: {e}")

        # IoU 매칭
        matched_gt = [False] * len(gt_objects)
        for gi, gt in enumerate(gt_objects):
            for pred in preds:
                if pred["label"] == gt["label_ours"] and \
                        _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    matched_gt[gi] = True
                    break

        all_results.append({
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "gt": gt_objects,
            "preds": preds,
            "matched_gt": matched_gt,
        })

        if (i + 1) % 50 == 0 or (i + 1) == n:
            done = sum(sum(r["matched_gt"]) for r in all_results)
            total_gt_so_far = sum(len(r["gt"]) for r in all_results)
            recall_pct = done / total_gt_so_far * 100 if total_gt_so_far else 0
            print(f"  [{i+1:>4}/{n}] 누적 Recall {recall_pct:.1f}%  "
                  f"(탐지 {done}/{total_gt_so_far})")

    print(f"\n[2단계 완료] 전체 {n}장 추론 완료")
    return all_results


# ── 3단계: 분석 ───────────────────────────────────────────────────────────────

def analyze(all_results: list[dict], output_dir: Path) -> None:
    total_gt  = sum(len(r["gt"]) for r in all_results)
    total_hit = sum(sum(r["matched_gt"]) for r in all_results)
    overall   = total_hit / total_gt if total_gt else 0.0

    cls_gt:  dict[str, int] = defaultdict(int)
    cls_hit: dict[str, int] = defaultdict(int)
    for r in all_results:
        for gi, gt in enumerate(r["gt"]):
            cls_gt[gt["label_ours"]] += 1
            if r["matched_gt"][gi]:
                cls_hit[gt["label_ours"]] += 1

    cls_recall = {c: cls_hit[c] / cls_gt[c] for c in cls_gt}
    cls_recall_asc = sorted(cls_recall.items(), key=lambda x: x[1])

    fp_cnt: dict[str, int] = defaultdict(int)
    for r in all_results:
        gt_labels = {gt["label_ours"] for gt in r["gt"]}
        for pred in r["preds"]:
            if pred["label"] not in gt_labels:
                fp_cnt[pred["label"]] += 1
    fp_top10 = sorted(fp_cnt.items(), key=lambda x: -x[1])[:10]

    rc_cluttered, rc_sparse = [], []
    for r in all_results:
        n_gt = len(r["gt"])
        if n_gt == 0:
            continue
        rv = sum(r["matched_gt"]) / n_gt
        if n_gt >= 5:
            rc_cluttered.append(rv)
        elif n_gt <= 3:
            rc_sparse.append(rv)

    avg_cl = sum(rc_cluttered) / len(rc_cluttered) if rc_cluttered else 0.0
    avg_sp = sum(rc_sparse) / len(rc_sparse) if rc_sparse else 0.0

    urgent = sorted(
        [(c, cls_recall[c], cls_gt[c]) for c in cls_recall if cls_gt[c] >= 5],
        key=lambda x: x[1]
    )[:5]

    # COCO 결과 로드 시도
    coco_recall_str = "미실행"
    coco_csv = _ROOT / "outputs" / "coco_eval" / "class_recall.csv"
    coco_recall_val = None
    if coco_csv.exists():
        import csv as csv_mod
        coco_rows: dict[str, float] = {}
        with open(coco_csv, encoding="utf-8-sig") as f:
            for row in csv_mod.DictReader(f):
                coco_rows[row["class"]] = float(row["recall"])
        # 전체 recall은 summary에서 읽거나 재계산 불가 → N/A
        coco_recall_str = "CSV 로드됨"

    SEP = "=" * 66
    lines = [
        SEP,
        "  Detecting Working Desk — YOLO-World Zero-shot 평가 결과",
        SEP,
        f"  평가 이미지:   {len(all_results)}장",
        f"  GT 객체 총수:  {total_gt}개",
        f"  탐지 성공:     {total_hit}개",
        "",
        f"(1) 전체 Recall: {overall*100:.1f}%",
        "",
        "(2) 클래스별 Recall (낮은 순):",
    ]
    for cls, rc in cls_recall_asc:
        lines.append(f"  {cls:<24} {rc*100:5.1f}% |{_bar(rc)}")

    lines += ["", "(3) 가장 많이 놓친 클래스 TOP 10:"]
    miss_sorted = sorted(
        [(c, cls_gt[c] - cls_hit[c], cls_gt[c]) for c in cls_gt],
        key=lambda x: -x[1]
    )[:10]
    for cls, missed, total in miss_sorted:
        lines.append(
            f"  {cls:<24}  놓침 {missed:>3}/{total:<3}  miss {missed/total*100:.1f}%"
        )

    lines += ["", "(4) False Positive TOP 10 (GT 없는데 예측한 클래스):"]
    for cls, cnt in fp_top10:
        lines.append(f"  {cls:<24}  {cnt}회")

    lines += [
        "",
        "(5) Cluttered scene 성능 비교:",
        f"  GT≥5 이미지 ({len(rc_cluttered)}장)  평균 Recall: {avg_cl*100:.1f}%",
        f"  GT≤3 이미지 ({len(rc_sparse)}장)  평균 Recall: {avg_sp*100:.1f}%",
        f"  → 혼잡도 성능 차이: {(avg_sp - avg_cl)*100:.1f}%p",
        "",
        "[Fine-tune 시급 클래스 TOP 5  (GT≥5 & recall 최저)]",
    ]
    for rank, (cls, rc, n_gt) in enumerate(urgent, 1):
        lines.append(f"  {rank}. {cls:<24}  Recall {rc*100:.1f}%  (GT {n_gt}개)")

    # (6) 3개 데이터셋 비교표
    common = sorted(set(cls_recall.keys()))
    lines += [
        "",
        "(6) 데이터셋 비교표",
        f"  {'지표':<28} {'LVIS':>8} {'Working Desk':>14}",
        f"  {'-'*52}",
        f"  {'전체 Recall':<28} {LVIS_RECALL:>7.1f}% {overall*100:>13.1f}%",
        f"  {'혼잡도 성능 차이(%p)':<28} {'N/A':>8} {(avg_sp-avg_cl)*100:>13.1f}%p",
        "",
        f"  {'클래스별 (공통)':<28} {'LVIS':>8} {'Working Desk':>14}",
        f"  {'-'*52}",
    ]
    lvis_cls_recall = {
        "bottle": 42.6, "cup": 26.4, "fork": 36.4,
        "keyboard": 50.0, "mouse": None, "remote control": 42.1,
        "scissors": None, "spoon": 28.6, "phone": 46.2,
    }
    for cls in sorted(common):
        lv = lvis_cls_recall.get(cls)
        lv_str = f"{lv:.1f}%" if lv is not None else "  N/A"
        wd_str = f"{cls_recall[cls]*100:.1f}%"
        lines.append(f"  {cls:<28} {lv_str:>8} {wd_str:>13}")

    lines.append(SEP)
    summary = "\n".join(lines)
    print("\n" + summary)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    missed_objects = [
        {**gt, "image_id": r["image_id"], "image_path": r["image_path"]}
        for r in all_results
        for gi, gt in enumerate(r["gt"])
        if not r["matched_gt"][gi]
    ]

    (output_dir / "eval_summary.txt").write_text(summary, encoding="utf-8")
    with open(output_dir / "missed_objects.json", "w", encoding="utf-8") as f:
        json.dump(missed_objects, f, ensure_ascii=False, indent=2)
    with open(output_dir / "class_recall.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "gt_count", "hit_count", "recall"])
        writer.writeheader()
        for cls, rc in cls_recall_asc:
            writer.writerow({"class": cls, "gt_count": cls_gt[cls],
                             "hit_count": cls_hit[cls], "recall": round(rc, 4)})

    print(f"\n[저장 완료]  {output_dir}/")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Working Desk Roboflow 데이터셋 평가")
    p.add_argument("--api-key", default="i6JLRbdc9uMSjXk4anbw")
    p.add_argument("--skip-download", action="store_true",
                   help="이미 다운로드된 경우 스킵")
    p.add_argument("--output", default="outputs/desk_eval")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_dir = _ROOT / "eval" / "data" / "desk"

    if not args.skip_download:
        try:
            ds_dir = download_dataset(args.api_key)
        except Exception as e:
            print(f"[다운로드 오류] {e}")
            if not ds_dir.exists():
                sys.exit(1)
            print(f"기존 경로 사용: {ds_dir}")
    else:
        print(f"다운로드 스킵. 경로: {ds_dir}")

    all_results = run_inference(ds_dir)

    with open(output_dir / "desk_predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    analyze(all_results, output_dir)


if __name__ == "__main__":
    main()
