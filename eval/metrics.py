"""
평가 지표 라이브러리 (eval/ 전용).

v3 우선순위:
  1순위 (CV 코어): mAP@0.5, mAP@0.5:0.95, Small-object AP, Recall, Precision, FP rate
  2순위:          describe 신뢰도 감사 → audit.py 참조
  3순위 (보조):   cost-weighted decision accuracy, plain accuracy, false-discard rate

사용 예:
  from eval.metrics import compute_map, compute_map_range, compute_cost_weighted_acc
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# ── IoU ────────────────────────────────────────────────────────────────────────

def iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── AP / mAP ───────────────────────────────────────────────────────────────────

def _ap_from_pr(precisions: list[float], recalls: list[float]) -> float:
    """COCO-style AP: area under monotone-interpolated PR curve."""
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 1, 0, -1):          # 단조 감소 보정
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]
    return float(sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in idx))


def _ap_for_class(
    preds: list[tuple],   # (image_id, confidence, bbox)
    gts:   list[tuple],   # (image_id, bbox)
    iou_thr: float,
) -> float:
    """단일 클래스 AP at given IoU threshold."""
    n_gt = len(gts)
    if n_gt == 0:
        return 0.0

    # GT를 이미지별로 인덱싱
    gts_by_img: dict[str, list[tuple[int, list]]] = defaultdict(list)
    for i, (img_id, bbox) in enumerate(gts):
        gts_by_img[img_id].append((i, bbox))

    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    matched: set[int] = set()
    tps, fps = [], []

    for img_id, _conf, pred_bbox in preds_sorted:
        best_iou, best_i = 0.0, -1
        for gt_i, gt_bbox in gts_by_img.get(img_id, []):
            if gt_i in matched:
                continue
            s = iou(pred_bbox, gt_bbox)
            if s > best_iou:
                best_iou, best_i = s, gt_i

        if best_iou >= iou_thr and best_i >= 0:
            tps.append(1); fps.append(0)
            matched.add(best_i)
        else:
            tps.append(0); fps.append(1)

    cum_tp = np.cumsum(tps, dtype=float)
    cum_fp = np.cumsum(fps, dtype=float)
    recalls    = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp)

    return _ap_from_pr(precisions.tolist(), recalls.tolist())


def compute_map(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thr:   float = 0.5,
    small_obj_ratio: float | None = None,
) -> dict:
    """
    mAP at a single IoU threshold.

    all_preds: {image_id: [{label, confidence, bbox:[x1,y1,x2,y2]}, ...]}
    all_gts:   {image_id: [{label, bbox:[x1,y1,x2,y2], ...}]}
    small_obj_ratio: GT box가 image_area * ratio 미만이면 "소형". None이면 전체 계산.

    Returns:
      {"mAP": float, "per_class": {cls: ap},
       "small_mAP": float | None, "small_per_class": {cls: ap} | None}
    """
    # 클래스별 pred/gt 수집
    pred_cls: dict[str, list] = defaultdict(list)
    gt_cls:   dict[str, list] = defaultdict(list)
    # 소형 객체 GT (image_area 필요 → image_id별 size를 별도 주입하기 어려우므로 bbox 절대 크기 기준)
    # 소형 = bbox 면적 < 이미지 면적 * ratio → 이미지 크기를 모를 때는 COCO 절대 기준(32²=1024) fallback
    SMALL_ABS_PX = 96 * 96  # 이미지 크기 정보 없을 때 fallback (≈ 96² px)
    gt_small_cls: dict[str, list] = defaultdict(list)

    for img_id, preds in all_preds.items():
        for p in preds:
            pred_cls[p["label"]].append((img_id, p["confidence"], p["bbox"]))

    for img_id, gts in all_gts.items():
        for g in gts:
            if g.get("decision") == "invalid":
                continue                              # 오탐 GT 제외
            gt_cls[g["label"]].append((img_id, g["bbox"]))
            bbox = g["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < SMALL_ABS_PX:
                gt_small_cls[g["label"]].append((img_id, g["bbox"]))

    def _compute(gt_dict: dict[str, list]) -> dict:
        aps: dict[str, float] = {}
        for cls, gts in gt_dict.items():
            preds = pred_cls.get(cls, [])
            aps[cls] = _ap_for_class(preds, gts, iou_thr)
        for cls in pred_cls:
            if cls not in aps:
                aps[cls] = 0.0  # GT 없는 클래스의 예측 = 모두 FP → AP=0
        map_val = float(np.mean(list(aps.values()))) if aps else 0.0
        return {"mAP": round(map_val, 4),
                "per_class": {k: round(v, 4) for k, v in sorted(aps.items())}}

    result = _compute(gt_cls)

    if gt_small_cls:
        small_result = _compute(gt_small_cls)
        result["small_mAP"]       = small_result["mAP"]
        result["small_per_class"] = small_result["per_class"]
    else:
        result["small_mAP"]       = None
        result["small_per_class"] = None

    return result


def compute_map_range(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thrs:  list[float] | None = None,
) -> float:
    """mAP@0.5:0.95 (COCO-style, 10개 threshold 평균)."""
    if iou_thrs is None:
        iou_thrs = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]
    maps = [compute_map(all_preds, all_gts, t)["mAP"] for t in iou_thrs]
    return round(float(np.mean(maps)), 4)


# ── Detection 지표 (per-image greedy, recall/precision/FP 등) ──────────────────

def detection_metrics_per_image(
    preds: list[dict],
    gt_objs: list[dict],
    iou_thr: float = 0.5,
    small_area_px: int = SMALL_ABS_PX if False else 96 * 96,
) -> dict:
    """
    이미지 1장 단위 greedy 매칭 기반 지표.
    preds:   [{label, confidence, bbox}, ...]
    gt_objs: [{label, bbox, ...}, ...]
    """
    gt_real = [o for o in gt_objs if o.get("decision") != "invalid"]

    matched_gt: set[int] = set()
    fp = 0
    for pred in sorted(preds, key=lambda x: x["confidence"], reverse=True):
        best_iou, best_i = 0.0, -1
        for i, gt in enumerate(gt_real):
            if i in matched_gt:
                continue
            s = iou(pred["bbox"], gt["bbox"])
            if s > best_iou:
                best_iou, best_i = s, i
        if best_iou >= iou_thr and best_i >= 0:
            matched_gt.add(best_i)
        else:
            fp += 1

    tp = len(matched_gt)
    fn = len(gt_real) - tp
    recall    = tp / len(gt_real) if gt_real else float("nan")
    precision = tp / (tp + fp)    if (tp + fp) > 0 else float("nan")

    small_gt = [o for o in gt_real
                if (o["bbox"][2]-o["bbox"][0]) * (o["bbox"][3]-o["bbox"][1]) < small_area_px]
    if small_gt:
        sm_matched = sum(
            1 for i, o in enumerate(gt_real)
            if i in matched_gt
            and (o["bbox"][2]-o["bbox"][0]) * (o["bbox"][3]-o["bbox"][1]) < small_area_px
        )
        small_recall = sm_matched / len(small_gt)
    else:
        small_recall = float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "recall":       round(recall, 4)       if not _isnan(recall) else float("nan"),
        "precision":    round(precision, 4)    if not _isnan(precision) else float("nan"),
        "small_gt":     len(small_gt),
        "small_recall": round(small_recall, 4) if not _isnan(small_recall) else float("nan"),
    }


def _isnan(v: float) -> bool:
    import math
    return math.isnan(v)


# ── Cost-weighted Decision Accuracy (보조 지표, v3 §7) ─────────────────────────

# v3 cost matrix (rows=GT, cols=예측): 비가역 오류(멀쩡한 걸 버리라 함)에 최대 가중
#               pred:discard  pred:relocate  pred:keep
# gt:discard         0             2            3       ← 쓰레기 방치
# gt:relocate        3             0            1
# gt:keep            4             1            0       ← 멀쩡한 것 버림 (최악)
COST_MATRIX: dict[str, dict[str, int]] = {
    "discard":  {"discard": 0, "relocate": 2, "keep": 3},
    "relocate": {"discard": 3, "relocate": 0, "keep": 1},
    "keep":     {"discard": 4, "relocate": 1, "keep": 0},
}
_MAX_COST = max(v for row in COST_MATRIX.values() for v in row.values())  # 4


def compute_cost_weighted_acc(
    predictions: list[str],
    gt_labels:   list[str],
    cost_matrix: dict[str, dict[str, int]] = COST_MATRIX,
    max_cost:    int | None = None,
) -> dict:
    """
    predictions: 예측 decision 리스트 ("discard"|"relocate"|"keep")
    gt_labels:   GT decision 리스트 (동일 순서, "invalid"는 자동 제외)
    Returns:
      cost_weighted_acc  = 1 - mean_normalized_cost (높을수록 좋음)
      plain_acc          = 단순 정확도
      false_discard_rate = GT=keep인데 pred=discard 비율 (비가역 오류)
      discard_recall     = GT=discard인데 pred=discard 비율 (쓰레기 포착률)
      n                  = 유효 쌍 수 (invalid 제외)
    """
    if max_cost is None:
        max_cost = max(v for row in cost_matrix.values() for v in row.values())

    valid_decisions = set(cost_matrix.keys())
    pairs = [
        (p, g) for p, g in zip(predictions, gt_labels)
        if g in valid_decisions and p in valid_decisions
    ]

    if not pairs:
        nan = float("nan")
        return {"cost_weighted_acc": nan, "plain_acc": nan,
                "false_discard_rate": nan, "discard_recall": nan, "n": 0}

    costs          = [cost_matrix[g][p] for p, g in pairs]
    plain_correct  = [int(p == g)                              for p, g in pairs]
    false_discards = [int(p == "discard" and g == "keep")      for p, g in pairs]
    gt_discards    = [g for _, g in pairs if g == "discard"]
    pred_discards_when_gt_discard = [
        int(p == "discard") for p, g in pairs if g == "discard"
    ]

    return {
        "cost_weighted_acc":  round(1.0 - sum(costs) / (len(costs) * max_cost), 4),
        "plain_acc":          round(sum(plain_correct) / len(pairs), 4),
        "false_discard_rate": round(sum(false_discards) / len(pairs), 4),
        "discard_recall":     round(
            sum(pred_discards_when_gt_discard) / len(gt_discards), 4
        ) if gt_discards else float("nan"),
        "n": len(pairs),
    }
