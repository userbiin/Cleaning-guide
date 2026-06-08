"""
Base detector selection — zero-shot 탐지 성능 비교 (v3 CV 코어 지표).

[GT 없이: 정성적 비교]
  python eval/compare_detectors.py --images test_images --detectors yolo-world

[GT 있을 때: mAP@0.5 / mAP@0.5:0.95 / Recall / Small-Object 정량 비교]
  python eval/compare_detectors.py \\
      --images eval/data/dev \\
      --gt eval/ground_truth \\
      --detectors yolo-world yoloe \\
      --output outputs/detector_comparison

GT JSON 형식 (eval/ground_truth/<stem>.json):
  {
    "image_id": "img001",
    "objects": [
      {"label": "bottle", "bbox": [x1, y1, x2, y2], "decision": "discard"},
      ...
    ]
  }
  decision="invalid" 객체는 오탐(FP)으로 탐지 지표 계산에서 제외됨.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 프로젝트 루트를 sys.path에 추가 (어느 디렉터리에서 실행해도 동작)
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from models.detector import get_detector
from eval.metrics import compute_map, compute_map_range

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

# COCO 기준 "소형" = 이미지 면적의 0.5% 이하
SMALL_OBJ_RATIO = 0.005


# ── IoU / matching ────────────────────────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match(preds: list[dict], gt_objs: list[dict], iou_thr: float) -> tuple[set, int, int]:
    """
    Greedy matching (confidence-descending).
    Returns: (matched_gt_indices, fp_count, fn_count)
    """
    matched_gt: set[int] = set()
    fp = 0
    for pred in sorted(preds, key=lambda x: x["confidence"], reverse=True):
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_objs):
            if i in matched_gt:
                continue
            s = _iou(pred["bbox"], gt["bbox"])
            if s > best_iou:
                best_iou, best_idx = s, i
        if best_iou >= iou_thr and best_idx >= 0:
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_objs) - len(matched_gt)
    return matched_gt, fp, fn


# ── 시각화 ────────────────────────────────────────────────────────────────────

_COLORS = {
    "yolo-world": (0, 200, 80),
    "yoloe":      (255, 140, 0),
    "default":    (100, 100, 255),
}


def _annotate(image_path: str, detections: list[dict],
              gt_objs: list[dict] | None, output_path: Path,
              detector_name: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        return
    color = _COLORS.get(detector_name, _COLORS["default"])

    # GT boxes (흰색 점선)
    if gt_objs:
        for gt in gt_objs:
            x1, y1, x2, y2 = gt["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(img, f"GT:{gt['label']}", (x1, max(y1-4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Predicted boxes
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d['label']} {d['confidence']:.2f}",
                    (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


# ── 메인 비교 로직 ─────────────────────────────────────────────────────────────

def run_comparison(image_dir: Path, detector_names: list[str],
                   gt_dir: Path | None, output_dir: Path,
                   conf: float, iou_thr: float) -> None:

    image_paths = sorted(p for p in image_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        print(f"[ERROR] No images found in {image_dir}"); return

    # GT 로드
    has_gt = gt_dir is not None and gt_dir.exists()
    gt_map: dict[str, list[dict]] = {}
    if has_gt:
        for gf in gt_dir.glob("*.json"):
            with open(gf, encoding="utf-8") as f:
                data = json.load(f)
            gt_map[gf.stem] = data.get("objects", [])
        print(f"GT 파일: {len(gt_map)}개 로드")
    else:
        print("GT 없음 → 정성적(시각) 비교 모드")

    # 탐지기 로드
    detectors: dict[str, object] = {}
    for name in detector_names:
        print(f"[{name}] 로딩...", end=" ", flush=True)
        try:
            detectors[name] = get_detector(name, conf=conf)
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")

    if not detectors:
        print("[ERROR] 사용 가능한 탐지기가 없습니다."); return

    rows = []
    # mAP 계산용: 이미지 전체에 걸쳐 pred/gt 축적
    all_preds_acc: dict[str, dict[str, list]] = {n: {} for n in detectors}
    all_gts_acc:   dict[str, list[dict]] = {}

    for img_path in image_paths:
        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            continue
        h, w = img_cv.shape[:2]
        img_area = h * w
        small_thr_px = img_area * SMALL_OBJ_RATIO

        gt_objs_all = gt_map.get(img_path.stem, []) if has_gt else []
        # "invalid" GT 제외 후 탐지 지표 계산에 사용
        gt_objs = [o for o in gt_objs_all if o.get("decision") != "invalid"]

        if gt_objs_all:
            all_gts_acc[img_path.stem] = gt_objs  # invalid 제외 버전 저장

        for det_name, detector in detectors.items():
            preds = detector.detect(str(img_path))
            all_preds_acc[det_name][img_path.stem] = preds  # mAP용 축적

            _annotate(
                str(img_path), preds, gt_objs if gt_objs else None,
                output_dir / det_name / f"{img_path.stem}_det.jpg",
                det_name,
            )

            row: dict = {
                "image":      img_path.name,
                "detector":   det_name,
                "pred_count": len(preds),
            }

            if gt_objs:
                matched, fp, fn = _match(preds, gt_objs, iou_thr)
                tp = len(matched)
                recall    = tp / len(gt_objs) if gt_objs else float("nan")
                precision = tp / (tp + fp)    if (tp + fp) > 0 else float("nan")

                # 소형 객체
                small_gt = [
                    o for o in gt_objs
                    if (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1])
                    < small_thr_px
                ]
                if small_gt:
                    sm_matched = sum(
                        1 for i, o in enumerate(gt_objs)
                        if i in matched
                        and (o["bbox"][2]-o["bbox"][0]) * (o["bbox"][3]-o["bbox"][1]) < small_thr_px
                    )
                    small_recall = sm_matched / len(small_gt)
                else:
                    small_recall = float("nan")

                row.update({
                    "gt_count":      len(gt_objs),
                    "tp": tp, "fp": fp, "fn": fn,
                    "recall":        round(recall, 3),
                    "precision":     round(precision, 3),
                    "small_gt":      len(small_gt),
                    "small_recall":  round(small_recall, 3) if not np.isnan(small_recall) else float("nan"),
                })
                status = (f"recall={recall:.2%}  precision={precision:.2%}"
                          f"  tp={tp} fp={fp} fn={fn}")
            else:
                status = f"탐지 {len(preds)}개"

            print(f"  {img_path.name} | {det_name}: {status}")
            rows.append(row)

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = output_dir / "detector_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n상세 결과: {csv_path}")

    # ── mAP 계산 (GT 있을 때) ──────────────────────────────────────────────────
    map_rows = []
    if has_gt and all_gts_acc:
        print("\nmAP 계산 중...")
        for det_name in detectors:
            preds_this = all_preds_acc[det_name]
            m50_result  = compute_map(preds_this, all_gts_acc, iou_thr=0.5)
            m5095       = compute_map_range(preds_this, all_gts_acc)
            small_map   = m50_result.get("small_mAP")
            map_rows.append({
                "detector":        det_name,
                "mAP@0.5":         m50_result["mAP"],
                "mAP@0.5:0.95":    m5095,
                "small_mAP@0.5":   small_map if small_map is not None else float("nan"),
            })
            print(f"  [{det_name}]  mAP@0.5={m50_result['mAP']:.4f}  "
                  f"mAP@0.5:0.95={m5095:.4f}"
                  + (f"  small_mAP={small_map:.4f}" if small_map else ""))

            # 클래스별 AP 저장
            ap_path = output_dir / f"{det_name}_per_class_ap.json"
            import json as _json
            ap_path.write_text(
                _json.dumps(m50_result["per_class"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        map_df = pd.DataFrame(map_rows).set_index("detector")
        map_csv = output_dir / "map_summary.csv"
        map_df.to_csv(map_csv, encoding="utf-8-sig")
        print(f"\nmAP 요약: {map_csv}")
    else:
        map_df = None

    # ── 탐지기별 요약 출력 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if has_gt and "recall" in df.columns:
        metric_cols = [c for c in ["recall", "precision", "small_recall", "tp", "fp", "fn"]
                       if c in df.columns]
        summary = df.groupby("detector")[metric_cols].mean().round(3)
        summary_path = output_dir / "summary.csv"
        summary.to_csv(summary_path, encoding="utf-8-sig")

        print("탐지기별 평균 — per-image greedy (recall/precision)")
        print(summary.to_string())
        if map_df is not None:
            print("\n탐지기별 mAP (전체 이미지 통합, COCO-style)")
            print(map_df.round(4).to_string())
        print(f"\n요약: {summary_path}")
        _print_winner(summary, map_df)
    else:
        print("탐지기별 탐지 수 통계 (GT 없음)")
        print(df.groupby("detector")["pred_count"].describe().round(1).to_string())

    print(f"\n어노테이션 이미지: {output_dir}")


def _print_winner(summary: pd.DataFrame, map_df: "pd.DataFrame | None" = None) -> None:
    """mAP@0.5 기준 최우수 탐지기 출력 (v3 CV 코어 지표)."""
    if map_df is not None and not map_df.empty and "mAP@0.5" in map_df.columns:
        best = map_df["mAP@0.5"].idxmax()
        print(f"\n★ mAP@0.5 기준 추천 base detector: [{best}]"
              f"  mAP@0.5={map_df.loc[best,'mAP@0.5']:.4f}"
              f"  mAP@0.5:0.95={map_df.loc[best,'mAP@0.5:0.95']:.4f}")
        if "small_mAP@0.5" in map_df.columns:
            best_small = map_df["small_mAP@0.5"].idxmax()
            print(f"  소형 객체 mAP 최고: [{best_small}]"
                  f"  small_mAP={map_df.loc[best_small,'small_mAP@0.5']:.4f}")
    elif "recall" in summary.columns and not summary.empty:
        best = summary["recall"].idxmax()
        print(f"\n★ Recall 기준 추천: [{best}]  recall={summary.loc[best,'recall']:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base detector selection via zero-shot recall comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--images", default="test_images",
                        help="이미지 디렉터리 (기본: test_images)")
    parser.add_argument("--gt", default=None,
                        help="GT JSON 디렉터리 (없으면 정성 비교만)")
    parser.add_argument("--detectors", nargs="+", default=["yolo-world"],
                        choices=["yolo-world", "yoloe"],
                        help="비교할 탐지기 (기본: yolo-world)")
    parser.add_argument("--output", default="outputs/detector_comparison",
                        help="결과 출력 디렉터리")
    parser.add_argument("--conf", type=float, default=0.08,
                        help="탐지 confidence threshold (기본: 0.08)")
    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="TP 판정용 IoU threshold (기본: 0.5)")
    args = parser.parse_args()

    run_comparison(
        image_dir=Path(args.images),
        detector_names=args.detectors,
        gt_dir=Path(args.gt) if args.gt else None,
        output_dir=Path(args.output),
        conf=args.conf,
        iou_thr=args.iou_thr,
    )


if __name__ == "__main__":
    main()
