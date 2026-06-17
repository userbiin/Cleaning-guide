"""
통합 데이터셋 YOLO-World fine-tune + 즉시 평가 — train_merged.py

실행:
  python train_merged.py
  python train_merged.py --eval-only   # 학습 건너뛰고 평가만 (best.pt 이미 있을 때)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

MERGED_DIR   = _ROOT / "datasets" / "merged"
BASELINE_CSV = _ROOT / "outputs" / "merged_val_baseline" / "per_class.csv"
OUT_DIR      = _ROOT / "outputs" / "merged_val_finetuned"
IOU_THR      = 0.5


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _load_gt(lbl_path: Path, W: int, H: int) -> list[dict]:
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
    ap = 0.0
    for t in [i / 100 for i in range(101)]:
        p = max((p for r, p in zip(recalls, precisions) if r >= t), default=0.0)
        ap += p / 101
    return ap


# ── 1단계: 학습 ───────────────────────────────────────────────────────────────

def train() -> Path:
    from ultralytics import YOLO

    model = YOLO("yolov8s-world.pt")
    result = model.train(
        data=str(MERGED_DIR / "data.yaml"),
        epochs=50,
        imgsz=1280,
        batch=8,
        device=0,

        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.1,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,

        project="runs/train",
        name="merged_v1",
    )

    best_pt = Path(result.save_dir) / "weights" / "best.pt"
    print(f"\n[학습 완료] best.pt: {best_pt}")
    return best_pt


# ── 2단계: val 추론 ───────────────────────────────────────────────────────────

def run_inference(model_path: Path, class_names: list[str]) -> list[dict]:
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[평가] {model_path.name}  device={device}  conf=0.08  imgsz=1280  iou=0.7")

    model = YOLO(str(model_path))
    # fine-tuned 모델도 class 이름을 명시적으로 지정
    try:
        model.set_classes(class_names)
    except Exception:
        pass  # 파인튜닝된 모델은 이미 클래스 내장

    img_dir = MERGED_DIR / "images" / "val"
    lbl_dir = MERGED_DIR / "labels" / "val"
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    n = len(img_paths)
    print(f"[평가] val {n}장")

    all_results = []
    for i, img_path in enumerate(img_paths):
        try:
            from PIL import Image as _Img
            img = _Img.open(img_path)
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
                        "conf": float(box.conf[0]),
                        "bbox": list(map(int, box.xyxy[0])),
                    })
        except Exception:
            preds = []

        all_results.append({"path": img_path, "gt": gt, "preds": preds})

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [{i+1:>4}/{n}]")

    return all_results


# ── 3단계: 메트릭 계산 ────────────────────────────────────────────────────────

def compute_metrics(all_results: list[dict], class_names: list[str]) -> dict:
    nc = len(class_names)
    per_cls_preds:    dict[int, list] = defaultdict(list)
    per_cls_gt_count: dict[int, int]  = defaultdict(int)
    cls_hit: dict[int, int] = defaultdict(int)
    cls_gt:  dict[int, int] = defaultdict(int)
    cls_fp:  dict[int, int] = defaultdict(int)

    for r in all_results:
        gt_matched = [False] * len(r["gt"])
        for g in r["gt"]:
            cls_gt[g["cls"]] += 1
            per_cls_gt_count[g["cls"]] += 1

        for pred in sorted(r["preds"], key=lambda x: -x["conf"]):
            pc = pred["cls"]
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(r["gt"]):
                if gt["cls"] != pc or gt_matched[gi]:
                    continue
                v = _iou(pred["bbox"], gt["bbox"])
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= IOU_THR:
                gt_matched[best_gi] = True
                cls_hit[pc] += 1
                per_cls_preds[pc].append((pred["conf"], True))
            else:
                cls_fp[pc] += 1
                per_cls_preds[pc].append((pred["conf"], False))

    ap_list = []
    for c in range(nc):
        preds_c = sorted(per_cls_preds[c], key=lambda x: -x[0])
        total_gt = per_cls_gt_count[c]
        if total_gt == 0:
            continue
        tp_cum = fp_cum = 0
        rs, ps = [], []
        for _, is_tp in preds_c:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            rs.append(tp_cum / total_gt)
            ps.append(tp_cum / (tp_cum + fp_cum))
        ap_list.append((c, _compute_ap(rs, ps)))

    total_gt_all  = sum(cls_gt.values())
    total_hit_all = sum(cls_hit.values())
    total_fp_all  = sum(cls_fp.values())
    total_pred    = total_hit_all + total_fp_all

    overall_recall    = total_hit_all / total_gt_all if total_gt_all else 0.0
    overall_precision = total_hit_all / total_pred   if total_pred   else 0.0
    mean_ap           = sum(ap for _, ap in ap_list) / len(ap_list) if ap_list else 0.0

    # Location Recall
    loc_hit = loc_total = 0
    for r in all_results:
        for gt in r["gt"]:
            loc_total += 1
            for pred in r["preds"]:
                if _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    loc_hit += 1
                    break
    loc_recall = loc_hit / loc_total if loc_total else 0.0

    # 혼동 행렬 (IoU>=0.5, 클래스 다른 쌍)
    mismatch: dict[tuple, int] = defaultdict(int)
    for r in all_results:
        gt_matched = [False] * len(r["gt"])
        for pred in sorted(r["preds"], key=lambda x: -x["conf"]):
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(r["gt"]):
                if gt_matched[gi]:
                    continue
                v = _iou(pred["bbox"], gt["bbox"])
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= IOU_THR and best_gi >= 0:
                gc = r["gt"][best_gi]["cls"]
                pc = pred["cls"]
                if gc != pc:
                    mismatch[(gc, pc)] += 1
                gt_matched[best_gi] = True

    # GT>=5 vs GT<=3
    dense_rc, sparse_rc = [], []
    for r in all_results:
        n_gt = len(r["gt"])
        if n_gt == 0:
            continue
        gm = [False] * n_gt
        for gi, gt in enumerate(r["gt"]):
            for pred in r["preds"]:
                if pred["cls"] == gt["cls"] and _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    gm[gi] = True
                    break
        rc = sum(gm) / n_gt
        if n_gt >= 5:
            dense_rc.append(rc)
        elif n_gt <= 3:
            sparse_rc.append(rc)

    return {
        "recall":     overall_recall,
        "precision":  overall_precision,
        "map50":      mean_ap,
        "loc_recall": loc_recall,
        "cls_recall": {c: cls_hit[c] / cls_gt[c] for c in cls_gt},
        "cls_gt":     dict(cls_gt),
        "cls_hit":    dict(cls_hit),
        "cls_fp":     dict(cls_fp),
        "ap_dict":    dict(ap_list),
        "mismatch":   dict(mismatch),
        "dense_avg":  sum(dense_rc) / len(dense_rc) if dense_rc else 0.0,
        "dense_n":    len(dense_rc),
        "sparse_avg": sum(sparse_rc) / len(sparse_rc) if sparse_rc else 0.0,
        "sparse_n":   len(sparse_rc),
        "total_gt":   total_gt_all,
        "total_hit":  total_hit_all,
        "n_images":   len(all_results),
    }


# ── 4단계: 비교 출력 + 저장 ──────────────────────────────────────────────────

def load_baseline() -> dict | None:
    if not BASELINE_CSV.exists():
        print(f"[경고] baseline CSV 없음: {BASELINE_CSV}")
        print("       먼저 python eval/eval_merged_val.py 를 실행하세요.")
        return None
    baseline = {}
    with open(BASELINE_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            baseline[row["class"]] = {
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "ap50": float(row["ap50"]),
                "gt": int(row["gt"]),
                "hit": int(row["hit"]),
            }
    # summary.txt에서 전체 지표 파싱
    summary_path = BASELINE_CSV.parent / "summary.txt"
    overall = {}
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("Recall:"):
                overall["recall"] = float(line.split()[1].rstrip("%")) / 100
            elif line.startswith("Precision:"):
                overall["precision"] = float(line.split()[1].rstrip("%")) / 100
            elif line.startswith("mAP@50:"):
                overall["map50"] = float(line.split()[1].rstrip("%")) / 100
            elif line.startswith("Location Recall:"):
                overall["loc_recall"] = float(line.split()[2].rstrip("%")) / 100
            elif "GT>=5" in line and "평균 Recall:" in line:
                overall["dense_avg"] = float(line.split("Recall:")[-1].strip().rstrip("%")) / 100
            elif "GT<=3" in line and "평균 Recall:" in line:
                overall["sparse_avg"] = float(line.split("Recall:")[-1].strip().rstrip("%")) / 100
    baseline["_overall"] = overall
    return baseline


def print_and_save(ft_metrics: dict, class_names: list[str], baseline: dict | None) -> None:
    SEP = "=" * 70
    enc = sys.stdout.encoding or "utf-8"

    def p(s: str) -> None:
        print(s.encode(enc, errors="replace").decode(enc))

    b = baseline or {}
    bov = b.get("_overall", {})

    lines = [
        SEP,
        "  YOLO-World fine-tune 후 val 평가 결과",
        SEP,
        f"  val 이미지: {ft_metrics['n_images']}장  |  GT: {ft_metrics['total_gt']}개",
        "",
        "[비교표]",
        f"  {'지표':<30} {'baseline (zero-shot)':>22} {'fine-tune 후':>14}  {'Delta':>8}",
        f"  {'-'*76}",
    ]

    def row(name, bval, fval, fmt="{:.1f}%"):
        bstr = fmt.format(bval * 100) if bval is not None else "N/A"
        fstr = fmt.format(fval * 100)
        delta = f"{(fval - bval)*100:+.1f}%p" if bval is not None else "N/A"
        lines.append(f"  {name:<30} {bstr:>22} {fstr:>14}  {delta:>8}")

    row("Recall",          bov.get("recall"),     ft_metrics["recall"])
    row("Precision",       bov.get("precision"),  ft_metrics["precision"])
    row("mAP@50",          bov.get("map50"),       ft_metrics["map50"])
    row("Location Recall", bov.get("loc_recall"),  ft_metrics["loc_recall"])
    row("GT>=5 avg Recall",bov.get("dense_avg"),   ft_metrics["dense_avg"])
    row("GT<=3 avg Recall",bov.get("sparse_avg"),  ft_metrics["sparse_avg"])

    lines += [
        "",
        "(2) 클래스별 Recall 비교 (fine-tune 후 낮은 순)",
        f"  {'클래스':<20} {'GT':>5} {'base':>8} {'fine':>8} {'Delta':>8}",
        f"  {'-'*52}",
    ]

    cls_recall_ft = [(c, v) for c, v in ft_metrics["cls_recall"].items()]
    cls_recall_ft.sort(key=lambda x: x[1])

    for c, rc_ft in cls_recall_ft:
        name   = class_names[c] if c < len(class_names) else str(c)
        gt_c   = ft_metrics["cls_gt"].get(c, 0)
        b_cls  = b.get(name, {})
        rc_b   = b_cls.get("recall")
        bstr   = f"{rc_b*100:.1f}%" if rc_b is not None else "N/A"
        delta  = f"{(rc_ft - rc_b)*100:+.1f}%p" if rc_b is not None else "N/A"
        lines.append(f"  {name:<20} {gt_c:>5} {bstr:>8} {rc_ft*100:>7.1f}%  {delta:>8}")

    lines += [
        "",
        "(3) 혼동 행렬 상위 10 (IoU>=0.5, 클래스 불일치)",
        f"  {'GT 클래스':<22} {'PRED 클래스':<22} {'횟수':>6}",
        f"  {'-'*52}",
    ]
    top10 = sorted(ft_metrics["mismatch"].items(), key=lambda x: -x[1])[:10]
    for (gc, pc), cnt in top10:
        gn = class_names[gc] if gc < len(class_names) else str(gc)
        pn = class_names[pc] if pc < len(class_names) else str(pc)
        lines.append(f"  {gn:<22} {pn:<22} {cnt:>6}")

    lines += [
        "",
        "(4) 이미지 밀도별 Recall",
        f"  GT>=5 ({ft_metrics['dense_n']}장): baseline {bov.get('dense_avg', 0)*100:.1f}%"
        f"  →  fine-tune {ft_metrics['dense_avg']*100:.1f}%"
        f"  ({(ft_metrics['dense_avg'] - bov.get('dense_avg',0))*100:+.1f}%p)",
        f"  GT<=3 ({ft_metrics['sparse_n']}장): baseline {bov.get('sparse_avg', 0)*100:.1f}%"
        f"  →  fine-tune {ft_metrics['sparse_avg']*100:.1f}%"
        f"  ({(ft_metrics['sparse_avg'] - bov.get('sparse_avg',0))*100:+.1f}%p)",
        SEP,
    ]

    summary = "\n".join(lines)
    for line in lines:
        p(line)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    with open(OUT_DIR / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "gt", "hit", "fp", "recall_base", "recall_ft", "delta", "ap50_ft"])
        ap_d = ft_metrics["ap_dict"]
        for c, rc_ft in sorted(ft_metrics["cls_recall"].items()):
            name  = class_names[c] if c < len(class_names) else str(c)
            rc_b  = b.get(name, {}).get("recall")
            delta = rc_ft - rc_b if rc_b is not None else None
            w.writerow([
                name,
                ft_metrics["cls_gt"].get(c, 0),
                ft_metrics["cls_hit"].get(c, 0),
                ft_metrics["cls_fp"].get(c, 0),
                f"{rc_b:.4f}" if rc_b is not None else "N/A",
                f"{rc_ft:.4f}",
                f"{delta:+.4f}" if delta is not None else "N/A",
                f"{ap_d.get(c, 0):.4f}",
            ])

    print(f"\n[저장 완료]")
    print(f"  {OUT_DIR}/summary.txt")
    print(f"  {OUT_DIR}/per_class.csv")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-only", action="store_true",
                   help="학습 없이 평가만 (runs/train/merged_v1/weights/best.pt 사용)")
    p.add_argument("--weights", type=str, default=None,
                   help="평가에 사용할 가중치 경로 (기본: runs/train/merged_v1/weights/best.pt)")
    args = p.parse_args()

    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    class_names: list[str] = meta["names"]
    print(f"클래스 ({len(class_names)}개): {class_names}")

    if args.eval_only:
        if args.weights:
            best_pt = Path(args.weights)
        else:
            best_pt = _ROOT / "runs" / "train" / "merged_v1" / "weights" / "best.pt"
        print(f"[학습 스킵] 가중치: {best_pt}")
    else:
        best_pt = train()

    if not best_pt.exists():
        print(f"[오류] best.pt 없음: {best_pt}")
        sys.exit(1)

    baseline = load_baseline()
    all_results = run_inference(best_pt, class_names)
    ft_metrics  = compute_metrics(all_results, class_names)
    print_and_save(ft_metrics, class_names, baseline)


if __name__ == "__main__":
    main()
