"""
COCO-2017 val zero-shot mAP 평가 — eval/eval_coco.py

FiftyOne으로 COCO-2017 validation 로드 후 우리 탐지기로 zero-shot 추론.
라벨 매핑(our → COCO-80)을 거쳐 pycocotools 기반 COCO mAP 산출.

실행 (RTX 4090 권장, ~5000장 전체):
  python eval/eval_coco.py

스모크 테스트 (200장 서브셋):
  python eval/eval_coco.py --max-samples 200

다른 탐지기 비교:
  python eval/eval_coco.py --detector yoloe --max-samples 200
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False

from PIL import Image

from models.detector import get_detector
from eval.metrics import compute_map, compute_map_range

# ── 라벨 매핑 ─────────────────────────────────────────────────────────────────
# 우리 OBJECT_CLASSES 레이블 → COCO-80 레이블 (탐지기 예측 저장 시 변환)
OUR_TO_COCO: dict[str, str] = {
    "scissors":       "scissors",
    "laptop":         "laptop",
    "keyboard":       "keyboard",
    "mouse":          "mouse",
    "phone":          "cell phone",
    "remote control": "remote",
    "cup":            "cup",
    "bottle":         "bottle",
    "bowl":           "bowl",
    "spoon":          "spoon",
    "fork":           "fork",
    "book":           "book",
    "backpack":       "backpack",
    "bag":            "handbag",
    "toothbrush":     "toothbrush",
    "monitor":        "tv",      # 근사 매핑 — TV 이미지 포함
    "mug": "cup",
}

COCO_EVAL_CLASSES: list[str] = sorted(set(OUR_TO_COCO.values()))
_OUR_MAPPED: set[str] = set(OUR_TO_COCO.keys())


# ── FiftyOne 데이터셋 로드 ─────────────────────────────────────────────────────

def _load_dataset(
    detector_name: str,
    max_samples: int | None,
    roboflow_dir: Path | None = None,
) -> "fo.Dataset":
    if roboflow_dir is not None:
        dataset_name = f"roboflow_desk_{detector_name}"
        if dataset_name in fo.list_datasets():
            print(f"기존 FiftyOne 데이터셋 로드: {dataset_name}")
            dataset = fo.load_dataset(dataset_name)
        else:
            print(f"Roboflow COCO 데이터셋 로드 중: {roboflow_dir}")
            dataset = fo.Dataset.from_dir(
                dataset_dir=str(roboflow_dir),
                dataset_type=fo.types.COCODetectionDataset,
                label_types=["detections"],
                dataset_name=dataset_name,
                persistent=True,
            )
    else:
        dataset_name = f"coco2017_val_orgdet_{detector_name}"
        if dataset_name in fo.list_datasets():
            print(f"기존 FiftyOne 데이터셋 로드: {dataset_name}")
            dataset = fo.load_dataset(dataset_name)
        else:
            print("COCO-2017 validation 다운로드/로드 중 (최초 실행 시 시간 소요)...")
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                label_types=["detections"],
                classes=COCO_EVAL_CLASSES,
                only_matching=True,
                dataset_name=dataset_name,
                persistent=True,
            )

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.take(max_samples, seed=42)
        print(f"서브셋 {max_samples}장 사용 (전체: 원래 데이터셋)")
    return dataset


# ── 메인 평가 루프 ─────────────────────────────────────────────────────────────

def run_eval(
    detector_name: str = "yolo-world",
    max_samples: int | None = None,
    conf: float = 0.08,
    output_dir: Path = Path("outputs/coco_eval"),
    roboflow_dir: Path | None = None,
) -> None:
    if not FIFTYONE_AVAILABLE:
        print("[ERROR] fiftyone 미설치. 설치: pip install fiftyone")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(detector_name, max_samples, roboflow_dir)
    n_samples = len(dataset)
    print(f"평가 이미지 수: {n_samples}  |  평가 클래스: {len(COCO_EVAL_CLASSES)}개")

    print(f"[{detector_name}] 탐지기 로딩...", end=" ", flush=True)
    detector = get_detector(detector_name, conf=conf)
    print("OK")

    pred_field = detector_name.replace("-", "_") + "_preds"
    eval_key   = "coco_eval_" + detector_name.replace("-", "_")

    # metrics.py 용 글로벌 축적 (our_label 기준)
    all_preds_acc: dict[str, list[dict]] = {}
    all_gts_acc:   dict[str, list[dict]] = {}

    print("\n추론 중 (각 이미지 → 탐지 → 라벨 변환 → FiftyOne 저장)...")
    for sample in dataset.iter_samples(progress=True, autosave=True):
        img_pil = Image.open(sample.filepath)
        w, h = img_pil.size
        img_id = sample.id

        raw_preds = detector.detect(sample.filepath)

        fo_detections = []
        pred_for_metrics: list[dict] = []

        for p in raw_preds:
            coco_lbl = OUR_TO_COCO.get(p["label"])
            if coco_lbl is None:
                continue
            x1, y1, x2, y2 = p["bbox"]
            # FiftyOne 정규화 bbox: [x, y, w, h] (0-1)
            fo_box = [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h]
            fo_detections.append(fo.Detection(
                label=coco_lbl,
                bounding_box=fo_box,
                confidence=p["confidence"],
            ))
            # metrics.py 용: COCO 라벨로 저장 (GT 라벨과 통일)
            pred_for_metrics.append({
                "label":      coco_lbl,
                "confidence": p["confidence"],
                "bbox":       p["bbox"],  # xyxy pixel
            })

        sample[pred_field] = fo.Detections(detections=fo_detections)
        all_preds_acc[img_id] = pred_for_metrics

        # GT도 COCO 라벨 기준으로 축적
        gt_fo = sample.ground_truth
        if gt_fo:
            gts_metric = []
            for det in gt_fo.detections:
                if det.label not in COCO_EVAL_CLASSES:
                    continue
                bx, by, bw, bh = det.bounding_box  # normalized
                gts_metric.append({
                    "label": det.label,
                    "bbox":  [int(bx*w), int(by*h),
                               int((bx+bw)*w), int((by+bh)*h)],
                })
            if gts_metric:
                all_gts_acc[img_id] = gts_metric

    # ── FiftyOne COCO mAP (0.5:0.95) ─────────────────────────────────────────
    print("\nFiftyOne COCO 평가 실행 중 (pycocotools)...")
    results = dataset.evaluate_detections(
        pred_field,
        gt_field="ground_truth",
        method="coco",
        compute_mAP=True,
        eval_key=eval_key,
        classes=COCO_EVAL_CLASSES,
    )
    map_5095 = results.mAP()

    # ── eval/metrics.py 기반 mAP@0.5 ─────────────────────────────────────────
    print("mAP@0.5 자체 계산 중...")
    m50_result = compute_map(all_preds_acc, all_gts_acc, iou_thr=0.5)
    map_50 = m50_result["mAP"]

    # ── 출력 ──────────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  detector:          {detector_name}")
    print(f"  eval images:       {n_samples}")
    print(f"  mAP@0.5:0.95      {map_5095:.4f}  (FiftyOne / pycocotools)")
    print(f"  mAP@0.5           {map_50:.4f}  (eval/metrics.py)")
    print(sep)
    print("\n[클래스별 COCO 평가 리포트]")
    results.print_report(classes=COCO_EVAL_CLASSES)

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    # per-class AP (from metrics.py @0.5)
    per_class_ap = m50_result["per_class"]
    ap_rows = [
        {
            "class":      cls,
            "AP@0.5":     per_class_ap.get(cls, 0.0),
        }
        for cls in sorted(COCO_EVAL_CLASSES)
    ]

    ap_csv = output_dir / f"{detector_name}_per_class_ap.csv"
    with open(ap_csv, "w", newline="", encoding="utf-8-sig") as f:
        w_csv = csv.DictWriter(f, fieldnames=["class", "AP@0.5"])
        w_csv.writeheader()
        w_csv.writerows(ap_rows)

    # 요약 JSON
    summary = {
        "detector":    detector_name,
        "n_images":    n_samples,
        "map_50_95":   round(map_5095, 4),
        "map_50":      round(map_50, 4),
        "per_class":   per_class_ap,
        "classes":     COCO_EVAL_CLASSES,
    }
    summary_json = output_dir / f"{detector_name}_coco_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n클래스별 AP CSV : {ap_csv}")
    print(f"요약 JSON        : {summary_json}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="COCO-2017 val zero-shot mAP 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--detector", default="yolo-world",
                   choices=["yolo-world", "yoloe"],
                   help="탐지기 이름 (기본: yolo-world)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="최대 이미지 수 — 스모크 테스트: 200 (기본: 전체)")
    p.add_argument("--conf", type=float, default=0.08,
                   help="탐지 confidence threshold (기본: 0.08)")
    p.add_argument("--output", default="outputs/coco_eval",
                   help="결과 저장 디렉터리 (기본: outputs/coco_eval)")
    p.add_argument("--roboflow-dir", default=None,
                   help="Roboflow COCO 데이터셋 경로 (예: eval/data/roboflow_desk/valid)")
    args = p.parse_args()
    run_eval(
        args.detector, args.max_samples, args.conf, Path(args.output),
        Path(args.roboflow_dir) if args.roboflow_dir else None,
    )


if __name__ == "__main__":
    main()
