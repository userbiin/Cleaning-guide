"""
통합 val 세트 zero-shot YOLO-World baseline 평가 — eval/eval_merged_val.py

실행:
  python eval/eval_merged_val.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

MERGED_DIR = _ROOT / "datasets" / "merged"
OUT_DIR    = _ROOT / "outputs" / "merged_val_baseline"
IOU_THR    = 0.5


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _load_gt(lbl_path: Path, W: int, H: int) -> list[dict]:
    """YOLO bbox 포맷 (cls cx cy w h 정규화) → xyxy 픽셀"""
    objects = []
    if not lbl_path.exists():
        return objects
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_idx = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        objects.append({"cls": cls_idx, "bbox": [x1, y1, x2, y2]})
    return objects


def _compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """101-point interpolation AP"""
    ap = 0.0
    for t in [i / 100 for i in range(101)]:
        p = max((p for r, p in zip(recalls, precisions) if r >= t), default=0.0)
        ap += p / 101
    return ap


# ── 추론 ─────────────────────────────────────────────────────────────────────

def run_inference(class_names: list[str]) -> list[dict]:
    import torch
    from ultralytics import YOLO
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[추론] device={device}  conf=0.08  imgsz=1280  iou=0.7")

    model = YOLO("yolov8s-world.pt")
    model.set_classes(class_names)

    img_dir = MERGED_DIR / "images" / "val"
    lbl_dir = MERGED_DIR / "labels" / "val"
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    n = len(img_paths)
    print(f"[추론] val 이미지 {n}장")

    all_results = []
    for i, img_path in enumerate(img_paths):
        try:
            img = Image.open(img_path)
            W, H = img.size
        except Exception:
            W, H = 640, 640

        gt = _load_gt(lbl_dir / (img_path.stem + ".txt"), W, H)

        try:
            res = model.predict(
                source=str(img_path),
                conf=0.08, imgsz=1280, iou=0.7,
                agnostic_nms=True, verbose=False, device=device,
            )
            raw = res[0]
            preds = []
            if raw.boxes is not None:
                for box in raw.boxes:
                    preds.append({
                        "cls":  int(box.cls[0]),
                        "label": raw.names[int(box.cls[0])],
                        "conf": float(box.conf[0]),
                        "bbox": list(map(int, box.xyxy[0])),
                    })
        except Exception as e:
            preds = []
            print(f"  [오류] {img_path.name}: {e}")

        all_results.append({"path": img_path, "gt": gt, "preds": preds, "W": W, "H": H})

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [{i+1:>4}/{n}] 완료")

    return all_results


# ── 분석 ─────────────────────────────────────────────────────────────────────

def analyze(all_results: list[dict], class_names: list[str]) -> None:
    nc = len(class_names)

    # ── (1) mAP@50, Precision, Recall 계산 ───────────────────────────────────
    # per-class: 예측을 confidence 내림차순 정렬 후 TP/FP 누적
    per_cls_preds: dict[int, list] = defaultdict(list)  # cls → [(conf, is_tp)]
    per_cls_gt_count: dict[int, int] = defaultdict(int)

    cls_hit:  dict[int, int] = defaultdict(int)
    cls_gt:   dict[int, int] = defaultdict(int)
    cls_fp:   dict[int, int] = defaultdict(int)

    for r in all_results:
        gt_matched = [False] * len(r["gt"])
        for g in r["gt"]:
            cls_gt[g["cls"]] += 1
            per_cls_gt_count[g["cls"]] += 1

        # confidence 내림차순 정렬
        preds_sorted = sorted(r["preds"], key=lambda x: -x["conf"])

        for pred in preds_sorted:
            pc = pred["cls"]
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(r["gt"]):
                if gt["cls"] != pc or gt_matched[gi]:
                    continue
                iou_val = _iou(pred["bbox"], gt["bbox"])
                if iou_val > best_iou:
                    best_iou, best_gi = iou_val, gi

            if best_iou >= IOU_THR:
                gt_matched[best_gi] = True
                cls_hit[pc] += 1
                per_cls_preds[pc].append((pred["conf"], True))
            else:
                cls_fp[pc] += 1
                per_cls_preds[pc].append((pred["conf"], False))

    # AP per class
    ap_list = []
    for c in range(nc):
        preds_c = sorted(per_cls_preds[c], key=lambda x: -x[0])
        total_gt = per_cls_gt_count[c]
        if total_gt == 0:
            continue
        tp_cum, fp_cum = 0, 0
        recalls, precisions = [], []
        for _, is_tp in preds_c:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            recalls.append(tp_cum / total_gt)
            precisions.append(tp_cum / (tp_cum + fp_cum))
        ap = _compute_ap(recalls, precisions)
        ap_list.append((c, ap))

    total_gt_all  = sum(cls_gt.values())
    total_hit_all = sum(cls_hit.values())
    total_fp_all  = sum(cls_fp.values())
    total_pred_all = total_hit_all + total_fp_all

    overall_recall    = total_hit_all / total_gt_all if total_gt_all else 0.0
    overall_precision = total_hit_all / total_pred_all if total_pred_all else 0.0
    mean_ap           = sum(ap for _, ap in ap_list) / len(ap_list) if ap_list else 0.0

    # ── (2) 클래스별 Recall ───────────────────────────────────────────────────
    cls_recall = {c: cls_hit[c] / cls_gt[c] for c in cls_gt}
    cls_recall_sorted = sorted(cls_recall.items(), key=lambda x: x[1])

    # ── (3) Location Recall ───────────────────────────────────────────────────
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

    # ── (4) 혼동 행렬 (IoU≥0.5인데 클래스 다른 쌍) ────────────────────────────
    mismatch_counts: dict[tuple, int] = defaultdict(int)
    for r in all_results:
        gt_matched = [False] * len(r["gt"])
        preds_sorted = sorted(r["preds"], key=lambda x: -x["conf"])
        for pred in preds_sorted:
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(r["gt"]):
                if gt_matched[gi]:
                    continue
                iou_val = _iou(pred["bbox"], gt["bbox"])
                if iou_val > best_iou:
                    best_iou, best_gi = iou_val, gi
            if best_iou >= IOU_THR and best_gi >= 0:
                gt_cls  = r["gt"][best_gi]["cls"]
                pred_cls = pred["cls"]
                if gt_cls != pred_cls:
                    mismatch_counts[(gt_cls, pred_cls)] += 1
                gt_matched[best_gi] = True

    top10_mismatch = sorted(mismatch_counts.items(), key=lambda x: -x[1])[:10]

    # ── (5) GT≥5 vs GT≤3 ─────────────────────────────────────────────────────
    dense_recalls, sparse_recalls = [], []
    for r in all_results:
        n_gt = len(r["gt"])
        if n_gt == 0:
            continue
        gt_matched = [False] * n_gt
        for gi, gt in enumerate(r["gt"]):
            for pred in r["preds"]:
                if pred["cls"] == gt["cls"] and _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    gt_matched[gi] = True
                    break
        img_rc = sum(gt_matched) / n_gt
        if n_gt >= 5:
            dense_recalls.append(img_rc)
        elif n_gt <= 3:
            sparse_recalls.append(img_rc)

    dense_avg  = sum(dense_recalls)  / len(dense_recalls)  if dense_recalls  else 0.0
    sparse_avg = sum(sparse_recalls) / len(sparse_recalls) if sparse_recalls else 0.0

    # ── 출력 ─────────────────────────────────────────────────────────────────
    SEP = "=" * 66
    lines = [
        SEP,
        "  통합 val zero-shot YOLO-World Baseline",
        SEP,
        f"  val 이미지:   {len(all_results)}장",
        f"  GT 객체 수:   {total_gt_all}개",
        "",
        "(1) 전체 지표",
        f"  Recall:     {overall_recall*100:.1f}%   ({total_hit_all}/{total_gt_all})",
        f"  Precision:  {overall_precision*100:.1f}%   ({total_hit_all}/{total_pred_all})",
        f"  mAP@50:     {mean_ap*100:.1f}%   ({len(ap_list)}개 클래스 평균)",
        "",
        "(2) 클래스별 Recall (낮은 순)",
        f"  {'클래스':<20} {'GT':>6} {'HIT':>6} {'Recall':>8}",
        f"  {'-'*44}",
    ]
    for c, rc in cls_recall_sorted:
        name = class_names[c] if c < len(class_names) else str(c)
        lines.append(f"  {name:<20} {cls_gt[c]:>6} {cls_hit[c]:>6} {rc*100:>7.1f}%  {'#'*int(rc*20)}")

    lines += [
        "",
        "(3) Location Recall (클래스 무관, IoU>=0.5)",
        f"  GT {loc_total}개 중 위치 탐지: {loc_hit}개",
        f"  Location Recall: {loc_recall*100:.1f}%",
        f"  클래스 Recall:   {overall_recall*100:.1f}%",
        f"  갭 (위치O 클래스X): {(loc_recall-overall_recall)*100:.1f}%p",
        "",
        "(4) 혼동 행렬 상위 10 (IoU>=0.5인데 클래스 다른 쌍)",
        f"  {'GT 클래스':<22} {'PRED 클래스':<22} {'횟수':>6}",
        f"  {'-'*52}",
    ]
    for (gc, pc), cnt in top10_mismatch:
        gn = class_names[gc] if gc < len(class_names) else str(gc)
        pn = class_names[pc] if pc < len(class_names) else str(pc)
        lines.append(f"  {gn:<22} {pn:<22} {cnt:>6}")

    lines += [
        "",
        "(5) 이미지 밀도별 Recall",
        f"  GT>=5 이미지 ({len(dense_recalls)}장)  평균 Recall: {dense_avg*100:.1f}%",
        f"  GT<=3 이미지 ({len(sparse_recalls)}장)  평균 Recall: {sparse_avg*100:.1f}%",
        f"  차이: {(dense_avg - sparse_avg)*100:.1f}%p",
        SEP,
    ]

    summary = "\n".join(lines)
    enc = sys.stdout.encoding or "utf-8"
    print(summary.encode(enc, errors="replace").decode(enc))

    # ── 파일 저장 ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    # per-class CSV
    with open(OUT_DIR / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "gt", "hit", "fp", "recall", "precision", "ap50"])
        ap_dict = dict(ap_list)
        for c in range(nc):
            if cls_gt[c] == 0 and cls_fp[c] == 0:
                continue
            name = class_names[c]
            gt_c   = cls_gt[c]
            hit_c  = cls_hit[c]
            fp_c   = cls_fp[c]
            rc     = hit_c / gt_c if gt_c else 0.0
            prec   = hit_c / (hit_c + fp_c) if (hit_c + fp_c) else 0.0
            ap     = ap_dict.get(c, 0.0)
            w.writerow([name, gt_c, hit_c, fp_c, f"{rc:.4f}", f"{prec:.4f}", f"{ap:.4f}"])

    # confusion top-20 CSV
    with open(OUT_DIR / "confusion_mismatch.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["gt_class", "pred_class", "count"])
        for (gc, pc), cnt in sorted(mismatch_counts.items(), key=lambda x: -x[1])[:20]:
            gn = class_names[gc] if gc < len(class_names) else str(gc)
            pn = class_names[pc] if pc < len(class_names) else str(pc)
            w.writerow([gn, pn, cnt])

    print(f"\n[저장 완료]")
    print(f"  {OUT_DIR}/summary.txt")
    print(f"  {OUT_DIR}/per_class.csv")
    print(f"  {OUT_DIR}/confusion_mismatch.csv")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    class_names: list[str] = meta["names"]
    print(f"클래스 ({len(class_names)}개): {class_names}")

    all_results = run_inference(class_names)
    analyze(all_results, class_names)


if __name__ == "__main__":
    main()
