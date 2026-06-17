"""
LVIS (HuggingFace) 기반 YOLO-World zero-shot 평가 — eval/eval_lvis.py

1단계: FiftyOne HuggingFace 연동으로 LVIS 500장 로드
2단계: YOLO-World 추론 (LVIS 클래스명 → OBJECT_CLASSES 매핑)
3단계: Recall / FP / cluttered scene 분석 및 결과 저장

실행:
  python eval/eval_lvis.py
  python eval/eval_lvis.py --max-samples 100   # 스모크 테스트
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.detector import OBJECT_CLASSES

# ── LVIS 클래스명 → OBJECT_CLASSES 매핑 ──────────────────────────────────────
# 실제 로드 후 누락 클래스는 1단계에서 자동 출력
LVIS_TO_OURS: dict[str, str] = {
    # 학습/문구
    "book":                "book",
    "notebook":            "notebook",
    "pen":                 "pen",
    "pencil":              "pencil",
    "eraser":              "eraser",
    "ruler":               "ruler",
    "scissors":            "scissors",
    # 전자기기
    "laptop_computer":     "laptop",
    "computer_keyboard":   "keyboard",
    "computer_mouse":      "mouse",
    "cellular_telephone":  "phone",
    "remote_control":      "remote control",
    "earphone":            "earphones",
    "charger":             "charger",
    "power_cord":          "cable",
    # 식기
    "cup":                 "cup",
    "mug":                 "mug",
    "bottle":              "bottle",
    "plate":               "plate",
    "bowl":                "bowl",
    "spoon":               "spoon",
    "fork":                "fork",
    # 의류
    "sock":                "socks",
    "shirt":               "shirt",
    "hat":                 "hat",
    "backpack":            "backpack",
    "bag":                 "bag",
    "shoe":                "shoe",
    # 생활용품
    "tissue":              "tissue",
    "toothbrush":          "toothbrush",
    "hairbrush":           "comb",
    "lipstick":            "cosmetics",
    "comb":                "comb",
    "mirror":              "mirror",
    # 기타
    "box":                 "box",
    "plastic_bag":         "plastic bag",
    "tin_can":             "trash",
    "wrapper":             "food wrapper",
}

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


# ── 1단계: HuggingFace에서 LVIS 로드 ─────────────────────────────────────────

def load_dataset(max_samples: int):
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh

    dataset_name = f"lvis_hf_{max_samples}"
    if dataset_name in fo.list_datasets():
        print(f"기존 FiftyOne 데이터셋 로드: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
    else:
        print("HuggingFace에서 Voxel51/LVIS 로드 중 (최초 실행 시 다운로드 발생)...")
        dataset = fouh.load_from_hub(
            "Voxel51/LVIS",
            max_samples=max_samples,
            name=dataset_name,
            persistent=True,
        )

    # 실제 필드명 탐색
    sample = dataset.first()
    print(f"\n[샘플 필드 확인]")
    print(f"  filepath: {sample.filepath}")
    print(f"  field_names: {sample.field_names}")

    # annotation 필드 자동 탐지 (ground_truth 또는 유사한 Detections 필드)
    gt_field = None
    for field in sample.field_names:
        val = sample.get_field(field)
        if hasattr(val, "detections"):
            gt_field = field
            break

    if gt_field is None:
        raise RuntimeError(
            f"Detections 필드를 찾지 못했습니다. 전체 필드: {sample.field_names}"
        )

    print(f"  GT 필드명: '{gt_field}'")
    print(f"  샘플 annotation 예시: {sample.get_field(gt_field)}")

    # 실제 클래스 목록 수집
    loaded_classes: set[str] = set()
    for s in dataset.iter_samples():
        ann = s.get_field(gt_field)
        if ann and ann.detections:
            for d in ann.detections:
                loaded_classes.add(d.label)

    # 매핑 커버리지 확인
    mapped = loaded_classes & set(LVIS_TO_OURS.keys())
    unmapped = loaded_classes - set(LVIS_TO_OURS.keys())

    print(f"\n[1단계 완료]")
    print(f"  총 이미지 수:        {len(dataset)}장")
    print(f"  전체 GT 클래스 수:   {len(loaded_classes)}개")
    print(f"  매핑된 클래스 수:    {len(mapped)}개  {sorted(mapped)}")
    if unmapped:
        print(f"  [미매핑 LVIS 클래스]: {sorted(unmapped)}")
        print(f"   → 이 클래스들은 GT에서 제외됩니다.")

    return dataset, gt_field


# ── 2단계: 추론 ───────────────────────────────────────────────────────────────

def run_inference(dataset, gt_field: str, conf: float) -> list[dict]:
    from ultralytics import YOLO
    from PIL import Image
    import torch

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cpu"
    )

    print(f"\n[2단계] YOLO-World 로딩... device={device}")
    model = YOLO("yolov8s-world.pt")
    model.set_classes(OBJECT_CLASSES + [""])
    print(f"  conf={conf}  imgsz=1280  iou=0.7  agnostic_nms=True")

    samples = list(dataset.iter_samples())
    n = len(samples)
    all_results = []

    for i, sample in enumerate(samples):
        img_path = sample.filepath
        try:
            img = Image.open(img_path)
            W, H = img.size
        except Exception:
            W, H = 1, 1  # 파일 읽기 실패 시 스킵

        # GT 추출
        gt_objects = []
        ann = sample.get_field(gt_field)
        if ann and ann.detections:
            for det in ann.detections:
                our_label = LVIS_TO_OURS.get(det.label)
                if our_label is None:
                    continue
                bx, by, bw, bh = det.bounding_box  # normalized [0,1]
                gt_objects.append({
                    "label_lvis": det.label,
                    "label_ours": our_label,
                    "bbox": [
                        int(bx * W), int(by * H),
                        int((bx + bw) * W), int((by + bh) * H),
                    ],
                })

        # 추론
        try:
            results = model.predict(
                source=img_path,
                conf=conf,
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
            print(f"  [추론 오류] {img_path}: {e}")

        # IoU 매칭으로 Recall 계산
        matched_gt = [False] * len(gt_objects)
        for gi, gt in enumerate(gt_objects):
            for pred in preds:
                if pred["label"] == gt["label_ours"] and \
                        _iou(gt["bbox"], pred["bbox"]) >= IOU_THR:
                    matched_gt[gi] = True
                    break

        all_results.append({
            "image_id": sample.id,
            "image_path": img_path,
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

    # FP: GT 라벨 없는데 예측된 클래스
    fp_cnt: dict[str, int] = defaultdict(int)
    for r in all_results:
        gt_labels = {gt["label_ours"] for gt in r["gt"]}
        for pred in r["preds"]:
            if pred["label"] not in gt_labels:
                fp_cnt[pred["label"]] += 1
    fp_top10 = sorted(fp_cnt.items(), key=lambda x: -x[1])[:10]

    # Cluttered vs sparse
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

    # Fine-tune 시급 (GT>=10, recall 낮은 순 top5)
    urgent = sorted(
        [(c, cls_recall[c], cls_gt[c]) for c in cls_recall if cls_gt[c] >= 10],
        key=lambda x: x[1]
    )[:5]

    # ── 텍스트 구성 ──────────────────────────────────────────────────────────
    SEP = "=" * 64
    lines = [
        SEP,
        "  LVIS (HuggingFace) — YOLO-World Zero-shot 평가 결과",
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
        "[Fine-tune 시급 클래스 TOP 5  (GT≥10 & recall 최저)]",
    ]
    for rank, (cls, rc, n_gt) in enumerate(urgent, 1):
        lines.append(
            f"  {rank}. {cls:<24}  Recall {rc*100:.1f}%  (GT {n_gt}개)"
        )
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
        writer = csv.DictWriter(
            f, fieldnames=["class", "gt_count", "hit_count", "recall"]
        )
        writer.writeheader()
        for cls, rc in cls_recall_asc:
            writer.writerow({
                "class": cls,
                "gt_count": cls_gt[cls],
                "hit_count": cls_hit[cls],
                "recall": round(rc, 4),
            })

    print(f"\n[저장 완료]")
    print(f"  {output_dir}/lvis_predictions.json")
    print(f"  {output_dir}/missed_objects.json")
    print(f"  {output_dir}/eval_summary.txt")
    print(f"  {output_dir}/class_recall.csv")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="LVIS HuggingFace YOLO-World 평가")
    p.add_argument("--max-samples", type=int, default=500,
                   help="다운로드할 최대 이미지 수 (기본: 500)")
    p.add_argument("--conf", type=float, default=0.08,
                   help="confidence threshold (기본: 0.08)")
    p.add_argument("--output", default="outputs/lvis_eval",
                   help="결과 저장 디렉터리 (기본: outputs/lvis_eval)")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1단계
    dataset, gt_field = load_dataset(args.max_samples)

    # 2단계
    all_results = run_inference(dataset, gt_field, args.conf)

    # predictions 저장
    pred_path = output_dir / "lvis_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # 3단계
    analyze(all_results, output_dir)


if __name__ == "__main__":
    main()
