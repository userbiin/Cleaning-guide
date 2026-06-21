"""
Zero-shot Detector 평가: YOLO-World / YOLOE / Florence-DRC
도메인(어질러진 책상) 라벨링 이미지 기준 Recall·정밀도·탐지율·mAP 비교.

Usage:
  python detector_eval.py                          # 기본: Downloads 경로
  python detector_eval.py --test-dir /path/to/test
  python detector_eval.py --detectors yolo-world yoloe  # Florence-DRC 제외
  python detector_eval.py --conf 0.1 --iou 0.5

출력 구성:
  PART 1 — 탐지기별 Recall / Precision / F1 / 탐지율 / mAP
  PART 2 — YOLO-World + Florence-DRC 앙상블 (추가 실험)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from models.detector import get_detector, OBJECT_CLASSES, build_prompt_list
from eval.metrics import compute_map, compute_map_range, iou as _iou_fn

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

# Roboflow export 경로 기본값 (사용자 환경에 맞게 변경)
_DEFAULT_TEST_DIR = Path.home() / "Downloads" / "cv.coco (1)" / "test"

# 데이터셋 내 오타 → 정규 라벨 매핑
_LABEL_FIXES: dict[str, str] = {
    "bootle":  "bottle",
    "ponch":   "pouch",
    "tumbler": "cup",
}

# OBJECT_CLASSES 기준 동의어 (detector 예측 라벨 정규화용)
_DETECTOR_SYNONYMS: dict[str, str] = {
    "mobilephone": "phone",
    "cellphone":   "phone",
    "mobile phone": "phone",
    "cell phone":  "phone",
    "laptop":      "laptop",
    "notebook":    "laptop",   # detector가 "notebook"을 laptop으로 탐지하는 경우
    "mug":         "cup",
    "tumbler":     "cup",
    "remote":      "remote controller",
    "remote control": "remote controller",
    "headphones":  "earphones",
    "earbuds":     "earphones",
    "glasses":     "sunglasses",
    "sunglass":    "sunglasses",
    "hairbrush":   "hair brush",
    "hair_brush":  "hair brush",
    "sticky note": "memo",
    "sticky_note": "memo",
    "note":        "memo",
    "pencilcase":  "pencil case",
    "pencil_case": "pencil case",
    "mousepad":    "mouse pad",
    "mouse_pad":   "mouse pad",
    "folder":      "folder",
    "binder":      "folder",
    "wire":        "cable",
    "cord":        "cable",
    "charger":     "charger",
    "adapter":     "charger",
    "earphone":    "earphones",
}


# ─── 라벨 정규화 ──────────────────────────────────────────────────────────────

def _norm(label: str) -> str:
    """공백·언더스코어 통일, 소문자화, 오타·동의어 치환."""
    s = label.strip().lower()
    s = re.sub(r"[\s_]+", " ", s)
    s = _LABEL_FIXES.get(s, s)
    return _DETECTOR_SYNONYMS.get(s, s)


# ─── COCO GT 로드 ─────────────────────────────────────────────────────────────

def load_coco_gt(coco_json: Path) -> dict[str, list[dict]]:
    """
    COCO 포맷 annotations → {image_stem: [{label, bbox:[x1,y1,x2,y2]}, ...]}
    bbox는 COCO [x, y, w, h] → [x1, y1, x2, y2] 변환.
    """
    with open(coco_json, encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: _norm(c["name"]) for c in data["categories"]}
    img_id_to_stem = {
        img["id"]: Path(img["file_name"]).stem for img in data["images"]
    }

    gt: dict[str, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        cat = cat_id_to_name[ann["category_id"]]
        if cat == "cv":   # root 카테고리 무시
            continue
        x, y, w, h = ann["bbox"]
        stem = img_id_to_stem[ann["image_id"]]
        gt[stem].append({
            "label": cat,
            "bbox":  [int(x), int(y), int(x + w), int(y + h)],
        })

    return dict(gt)


# ─── Florence-DRC Detector ────────────────────────────────────────────────────

class FlorenceDRCDetector:
    """
    Florence-2 <DENSE_REGION_CAPTION> 기반 open-vocab 탐지기.
    출력된 caption을 OBJECT_CLASSES 키워드와 매칭해 label 할당.
    confidence 점수는 미지원 → 전체 0.5 고정.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, PretrainedConfig

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        # 최신 transformers 호환 workaround (기존 describer.py 와 동일)
        if not hasattr(PretrainedConfig, "forced_bos_token_id"):
            PretrainedConfig.forced_bos_token_id = None

        print(f"[Florence-DRC] {model_id} 로드 중 (device={device})...", end=" ", flush=True)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        attn = "sdpa" if device == "cuda" else "eager"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, config=config, torch_dtype=self.dtype,
                attn_implementation=attn, trust_remote_code=True,
            ).eval().to(device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, config=config, torch_dtype=self.dtype,
                trust_remote_code=True,
            ).eval().to(device)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._classes_sorted = sorted(OBJECT_CLASSES, key=len, reverse=True)
        print("OK")

    def detect(self, image_path: str) -> list[dict]:
        import torch
        from PIL import Image

        pil = Image.open(image_path).convert("RGB")
        W, H = pil.size
        task = "<DENSE_REGION_CAPTION>"
        inputs = self.processor(text=task, images=pil, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device, self.dtype)
        ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=ids, pixel_values=pv,
                max_new_tokens=1024, num_beams=1,
                do_sample=False, use_cache=False,
            )

        gen_text = self.processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(gen_text, task=task, image_size=(W, H))
        drc = result.get("<DENSE_REGION_CAPTION>", {})

        out = []
        for bbox, cap in zip(drc.get("bboxes", []), drc.get("labels", [])):
            label = self._match_class(cap)
            if label is None:
                continue
            x1, y1, x2, y2 = bbox
            out.append({
                "label":      label,
                "confidence": 0.5,
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            })
        return out

    def _match_class(self, caption: str) -> str | None:
        """Caption에서 OBJECT_CLASSES 키워드 찾기 (긴 클래스명 우선)."""
        c = caption.lower()
        for cls in self._classes_sorted:
            if cls.lower() in c:
                return cls
        return None


# ─── 매칭 / NMS 유틸 ──────────────────────────────────────────────────────────

def _greedy_match(
    preds: list[dict],
    gt_objs: list[dict],
    iou_thr: float,
) -> tuple[int, int, int]:
    """신뢰도 내림차순 greedy 매칭. (tp, fp, fn) 반환."""
    matched_gt, fp_labels = _greedy_match_full(preds, gt_objs, iou_thr)
    tp = len(matched_gt)
    fn = len(gt_objs) - tp
    return tp, len(fp_labels), fn


def _greedy_match_full(
    preds: list[dict],
    gt_objs: list[dict],
    iou_thr: float,
) -> tuple[set[int], list[str]]:
    """
    신뢰도 내림차순 greedy 매칭.
    Returns: (matched_gt_indices, unmatched_pred_labels)
    클래스별 TP/FP/FN 추적에 사용.
    """
    matched_gt: set[int] = set()
    unmatched_pred_labels: list[str] = []

    for pred in sorted(preds, key=lambda x: x["confidence"], reverse=True):
        best_iou, best_i = 0.0, -1
        for i, gt in enumerate(gt_objs):
            if i in matched_gt:
                continue
            if pred["label"] != gt["label"]:
                continue
            s = _iou_fn(pred["bbox"], gt["bbox"])
            if s > best_iou:
                best_iou, best_i = s, i
        if best_iou >= iou_thr and best_i >= 0:
            matched_gt.add(best_i)
        else:
            unmatched_pred_labels.append(pred["label"])

    return matched_gt, unmatched_pred_labels


def _nms(detections: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """IoU 기반 NMS. 신뢰도 높은 박스 우선 유지."""
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept: list[dict] = []
    for d in sorted_dets:
        suppressed = any(_iou_fn(d["bbox"], k["bbox"]) > iou_thr for k in kept)
        if not suppressed:
            kept.append(d)
    return kept


# ─── 지표 출력 헬퍼 ───────────────────────────────────────────────────────────

def _print_sep(char: str = "=", width: int = 68) -> None:
    print(char * width)


def _print_metrics_block(
    name: str,
    tp: int, fp: int, fn: int,
    detected_imgs: int, total_imgs: int,
) -> None:
    n_gt = tp + fn
    recall    = tp / n_gt        if n_gt          else 0.0
    precision = tp / (tp + fp)   if (tp + fp)     else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    det_rate  = detected_imgs / total_imgs if total_imgs else 0.0

    print(f"\n[{name}]")
    print(f"  Recall     (탐지율)  : {recall:.1%}   ({tp} / {n_gt} GT 객체)")
    print(f"  Precision            : {precision:.1%}   ({tp} / {tp + fp} 예측)")
    print(f"  F1                   : {f1:.1%}")
    print(f"  이미지 탐지율        : {det_rate:.1%}   ({detected_imgs} / {total_imgs} 이미지)")
    print(f"  TP={tp}  FP={fp}  FN={fn}")


# ─── 클래스별 분석 출력 ──────────────────────────────────────────────────────

def _print_class_breakdown(
    det_name: str,
    class_gt: dict[str, int],
    class_tp: dict[str, int],
    class_fp: dict[str, int],
) -> None:
    """클래스별 GT / TP / FN / FP / Recall 테이블 출력 (FN 많은 순)."""
    rows = []
    all_classes = set(class_gt) | set(class_fp)
    for cls in all_classes:
        gt  = class_gt.get(cls, 0)
        tp  = class_tp.get(cls, 0)
        fp  = class_fp.get(cls, 0)
        fn  = gt - tp
        rec = tp / gt if gt > 0 else float("nan")
        rows.append((cls, gt, tp, fn, fp, rec))

    # GT 있는 클래스: FN 많은 순 → 탐지 못한 클래스 우선
    has_gt  = sorted([r for r in rows if r[1] > 0], key=lambda x: -x[3])
    no_gt   = sorted([r for r in rows if r[1] == 0], key=lambda x: -x[4])

    print(f"\n  [{det_name}] 클래스별 탐지 현황 (FN 많은 순)")
    print(f"  {'클래스':<22} {'GT':>5} {'TP':>5} {'FN':>5} {'FP':>5}  Recall")
    print("  " + "-" * 54)

    for cls, gt, tp, fn, fp, rec in has_gt:
        rec_str  = f"{rec:>6.1%}" if not (isinstance(rec, float) and rec != rec) else "   N/A"
        zero_tag = "  ← 전혀 미탐지" if tp == 0 and gt > 0 else ""
        print(f"  {cls:<22} {gt:>5} {tp:>5} {fn:>5} {fp:>5}  {rec_str}{zero_tag}")

    if no_gt:
        print(f"  {'── GT 없는 클래스 (FP만 발생) ──'}")
        for cls, gt, tp, fn, fp, rec in no_gt:
            print(f"  {cls:<22} {gt:>5}     -     -  {fp:>5}      -")


# ─── PART 1: 탐지기별 단독 평가 ──────────────────────────────────────────────

def run_part1(
    detector_factories: dict[str, "callable"],
    image_paths: list[Path],
    gt_map: dict[str, list[dict]],
    iou_thr: float,
) -> dict[str, dict]:
    """
    각 탐지기를 순차 로드→추론→해제해 MPS 메모리 충돌 방지.
    Returns: {det_name: {stem: {"preds": [...], "gt": [...]}}}
    """
    _print_sep()
    print("  PART 1: Zero-shot Detector 비교")
    print(f"  탐지기: {', '.join(detector_factories.keys())}")
    print(f"  이미지: {len(image_paths)}장  |  IoU threshold: {iou_thr}")
    _print_sep()

    all_results: dict[str, dict] = {}

    for det_name, factory in detector_factories.items():
        print(f"\n  [{det_name}] 로딩...", end=" ", flush=True)
        try:
            detector = factory()
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")
            continue

        det_res: dict[str, dict] = {}
        print(f"  [{det_name}] 추론 중 ({len(image_paths)}장)...")
        for img_path in image_paths:
            stem    = img_path.stem
            gt_objs = gt_map.get(stem, [])
            try:
                raw   = detector.detect(str(img_path))
                preds = [{**p, "label": _norm(p["label"])} for p in raw]
            except Exception as e:
                print(f"    [WARN] {img_path.name}: {e}")
                preds = []
            det_res[stem] = {"preds": preds, "gt": gt_objs}
            print(f"    {img_path.name[:44]:<44}  GT={len(gt_objs):3d}  preds={len(preds)}")

        all_results[det_name] = det_res

        # 다음 모델 로드 전 MPS 메모리 해제
        del detector
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # ── 지표 집계 ────────────────────────────────────────────────────────────
    print()
    _print_sep("-")
    print("  탐지기별 종합 지표 (IoU@{:.2f})".format(iou_thr))
    _print_sep("-")

    map_results: dict[str, dict] = {}

    for det_name in all_results:
        res = all_results[det_name]
        total_tp = total_fp = total_fn = 0
        detected_imgs = 0
        imgs_with_gt = 0
        all_preds_for_map: dict[str, list[dict]] = {}
        all_gts_for_map:   dict[str, list[dict]] = {}

        # 클래스별 추적
        class_gt: dict[str, int] = defaultdict(int)
        class_tp: dict[str, int] = defaultdict(int)
        class_fp: dict[str, int] = defaultdict(int)

        for stem, r in res.items():
            preds   = r["preds"]
            gt_objs = r["gt"]
            all_preds_for_map[stem] = preds

            if not gt_objs:
                continue
            all_gts_for_map[stem] = gt_objs
            imgs_with_gt += 1

            for gt in gt_objs:
                class_gt[gt["label"]] += 1

            matched_gt, fp_labels = _greedy_match_full(preds, gt_objs, iou_thr)
            tp  = len(matched_gt)
            fp  = len(fp_labels)
            fn  = len(gt_objs) - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if tp > 0:
                detected_imgs += 1

            for i, gt in enumerate(gt_objs):
                if i in matched_gt:
                    class_tp[gt["label"]] += 1
            for label in fp_labels:
                class_fp[label] += 1

        _print_metrics_block(det_name, total_tp, total_fp, total_fn,
                             detected_imgs, imgs_with_gt)

        # mAP 계산
        try:
            m50    = compute_map(all_preds_for_map, all_gts_for_map, iou_thr=0.5)
            m5095  = compute_map_range(all_preds_for_map, all_gts_for_map)
            small  = m50.get("small_mAP")
            map_results[det_name] = {"mAP@0.5": m50["mAP"], "mAP@0.5:0.95": m5095,
                                     "small_mAP@0.5": small}
            print(f"  mAP@0.5        : {m50['mAP']:.4f}")
            print(f"  mAP@0.5:0.95   : {m5095:.4f}")
            if small is not None:
                print(f"  small_mAP@0.5  : {small:.4f}")
        except Exception as e:
            print(f"  mAP 계산 실패: {e}")

        # 클래스별 분석
        _print_class_breakdown(det_name, class_gt, class_tp, class_fp)

    # ── 추천 탐지기 ──────────────────────────────────────────────────────────
    print()
    _print_sep("-")
    if map_results:
        best = max(map_results, key=lambda n: map_results[n].get("mAP@0.5", 0))
        m = map_results[best]
        print(f"  >> mAP@0.5 기준 추천 base detector: [{best}]"
              f"  mAP@0.5={m['mAP@0.5']:.4f}"
              f"  mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}")
    _print_sep("-")

    return all_results


# ─── PART 2: YOLO-World + Florence-DRC 앙상블 ────────────────────────────────

def run_part2(
    all_results: dict[str, dict],
    iou_thr: float,
) -> None:
    """
    YOLO-World 탐지 + Florence-DRC 탐지를 Union+NMS로 합친 뒤 Recall 변화 측정.
    YOLO-World 단독 대비 얼마나 개선되는지 보여줌.
    """
    print()
    _print_sep()
    print("  PART 2: YOLO-World + Florence-DRC 앙상블 (추가 실험)")
    print("  [앙상블 방식] Union after NMS(IoU={:.2f})".format(iou_thr))
    print("  [목적] Florence-DRC가 YOLO-World 누락 객체를 보완하는지 확인")
    _print_sep()

    if "yolo-world" not in all_results or "florence-drc" not in all_results:
        missing = [n for n in ["yolo-world", "florence-drc"] if n not in all_results]
        print(f"  [SKIP] 앙상블에 필요한 탐지기 없음: {missing}")
        return

    yw_res  = all_results["yolo-world"]
    fl_res  = all_results["florence-drc"]
    stems   = list(yw_res.keys())

    yw_tp_total = yw_fp_total = yw_fn_total = 0
    ens_tp_total = ens_fp_total = ens_fn_total = 0
    yw_det_imgs = ens_det_imgs = 0
    imgs_with_gt = 0
    fl_only_tp = 0   # Florence만 탐지한 TP 수

    print(f"\n{'이미지':<42} {'GT':>4} {'YW':>4} {'FL':>4} {'앙상블':>5}  YW_rec  ENS_rec")
    print("-" * 75)

    for stem in stems:
        yw_preds  = yw_res[stem]["preds"]
        fl_preds  = fl_res[stem]["preds"]
        gt_objs   = yw_res[stem]["gt"]

        if not gt_objs:
            continue
        imgs_with_gt += 1

        # 앙상블: 두 탐지기 합친 후 NMS
        ensemble_preds = _nms(yw_preds + fl_preds, iou_thr=iou_thr)

        yw_tp, yw_fp, yw_fn = _greedy_match(yw_preds, gt_objs, iou_thr)
        ens_tp, ens_fp, ens_fn = _greedy_match(ensemble_preds, gt_objs, iou_thr)

        yw_tp_total  += yw_tp;  yw_fp_total  += yw_fp;  yw_fn_total  += yw_fn
        ens_tp_total += ens_tp; ens_fp_total += ens_fp; ens_fn_total += ens_fn

        added = ens_tp - yw_tp
        fl_only_tp += max(0, added)

        if yw_tp > 0:  yw_det_imgs  += 1
        if ens_tp > 0: ens_det_imgs += 1

        n_gt    = len(gt_objs)
        yw_rec  = yw_tp / n_gt
        ens_rec = ens_tp / n_gt

        name = stem[:40]
        print(f"  {name:<40}  {n_gt:>4} {len(yw_preds):>4} {len(fl_preds):>4}"
              f" {len(ensemble_preds):>5}  {yw_rec:>6.1%}  {ens_rec:>6.1%}")

    # ── 집계 지표 ─────────────────────────────────────────────────────────────
    n_gt_total = yw_tp_total + yw_fn_total

    yw_recall  = yw_tp_total  / n_gt_total if n_gt_total else 0
    ens_recall = ens_tp_total / n_gt_total if n_gt_total else 0
    yw_prec    = yw_tp_total  / (yw_tp_total  + yw_fp_total)  if (yw_tp_total  + yw_fp_total)  else 0
    ens_prec   = ens_tp_total / (ens_tp_total + ens_fp_total) if (ens_tp_total + ens_fp_total) else 0

    print()
    _print_sep("-")
    print(f"  {'지표':<22} {'YOLO-World 단독':>16}  {'앙상블':>12}  {'변화':>8}")
    _print_sep("-")
    print(f"  {'Recall (탐지율)':<22} {yw_recall:>15.1%}  {ens_recall:>11.1%}  "
          f"  {(ens_recall - yw_recall):>+.1%}")
    print(f"  {'Precision':<22} {yw_prec:>15.1%}  {ens_prec:>11.1%}  "
          f"  {(ens_prec - yw_prec):>+.1%}")
    print(f"  {'TP':<22} {yw_tp_total:>16}  {ens_tp_total:>12}  "
          f"  {ens_tp_total - yw_tp_total:>+d}")
    print(f"  {'FP':<22} {yw_fp_total:>16}  {ens_fp_total:>12}  "
          f"  {ens_fp_total - yw_fp_total:>+d}")
    print(f"  {'FN':<22} {yw_fn_total:>16}  {ens_fn_total:>12}  "
          f"  {ens_fn_total - yw_fn_total:>+d}")
    _print_sep("-")
    print(f"  Florence-DRC 기여 TP   : {fl_only_tp}개")
    print(f"  이미지 탐지율 변화     : {yw_det_imgs}/{imgs_with_gt} → {ens_det_imgs}/{imgs_with_gt}")
    _print_sep("-")

    if ens_recall > yw_recall:
        gain = ens_recall - yw_recall
        print(f"  >> 앙상블이 YOLO-World 대비 Recall {gain:.1%}p 향상")
    else:
        print(f"  >> 앙상블 개선 없음 (Florence-DRC가 추가 TP를 제공하지 못함)")


# ─── PART 3: SAHI 슬라이싱 추론 ─────────────────────────────────────────────

def _sahi_inference(
    image_paths: list[Path],
    gt_map: dict[str, list[dict]],
    conf: float = 0.08,
    iou_thr: float = 0.5,
    slice_hw: int = 512,
    overlap: float = 0.2,
    rich_vocab: bool = False,
) -> "dict[str, dict] | None":
    """
    SAHI get_sliced_prediction으로 YOLO-World 추론.
    rich_vocab=True 이면 PROMPT_VOCAB 확장 프롬프트 사용 후 canonical로 역매핑.
    Returns: {stem: {"preds": [...], "gt": [...]}}  or  None (sahi 미설치)
    """
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        print("  [SKIP] sahi 미설치 → pip install sahi 후 재실행")
        return None

    import torch
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[SAHI] YOLO-World 로드 중 (conf={conf}, device={device})...", end=" ", flush=True)
    try:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolov8s-world.pt",
            confidence_threshold=conf,
            device=device,
        )
        if rich_vocab:
            prompts, canonical_map = build_prompt_list()
            classes = prompts + [""]
        else:
            canonical_map = {}
            classes = OBJECT_CLASSES + [""]
        sahi_model.model.set_classes(classes)
        # set_classes가 model.names를 업데이트해도 SAHI의 category_mapping은
        # 로드 시점의 COCO 이름으로 고정된다 → 수동으로 동기화 필요
        sahi_model.category_mapping = {str(i): name for i, name in enumerate(classes)}
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        return None

    results: dict[str, dict] = {}
    for img_path in image_paths:
        stem    = img_path.stem
        gt_objs = gt_map.get(stem, [])
        try:
            result = get_sliced_prediction(
                str(img_path), sahi_model,
                slice_height=slice_hw, slice_width=slice_hw,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                postprocess_type="NMS", postprocess_match_threshold=iou_thr,
                verbose=0,
            )
            preds = []
            for o in result.object_prediction_list:
                raw = o.category.name
                if raw == "":
                    continue
                label = _norm(canonical_map.get(raw, raw))
                preds.append({
                    "label":      label,
                    "confidence": o.score.value,
                    "bbox":       [o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy],
                })
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")
            preds = []

        results[stem] = {"preds": preds, "gt": gt_objs}
        print(f"  {img_path.name[:42]:<42}  GT={len(gt_objs):3d}  SAHI={len(preds)}")

    return results


def _summary_metrics(
    res: dict[str, dict],
    iou_thr: float,
    conf_filter: float | None = None,
) -> dict:
    """
    res = {stem: {"preds": [...], "gt": [...]}} 에서 통합 지표 계산.
    conf_filter: None이면 전체, 값 지정 시 해당 conf 이상 예측만 사용.
    """
    all_preds: dict[str, list[dict]] = {}
    all_gts:   dict[str, list[dict]] = {}
    total_tp = total_fp = total_fn = 0

    for stem, r in res.items():
        preds   = r["preds"]
        gt_objs = r["gt"]
        if conf_filter is not None:
            preds = [p for p in preds if p["confidence"] >= conf_filter]
        all_preds[stem] = preds
        if not gt_objs:
            continue
        all_gts[stem] = gt_objs
        tp, fp, fn = _greedy_match(preds, gt_objs, iou_thr)
        total_tp += tp; total_fp += fp; total_fn += fn

    n_gt      = total_tp + total_fn
    recall    = total_tp / n_gt if n_gt else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0

    try:
        m50   = compute_map(all_preds, all_gts, iou_thr=0.5)
        m5095 = compute_map_range(all_preds, all_gts)
        small = m50.get("small_mAP")
    except Exception:
        m50 = {"mAP": float("nan")}; m5095 = float("nan"); small = None

    return {
        "recall": recall, "precision": precision, "f1": f1,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "mAP@0.5": m50["mAP"], "mAP@0.5:0.95": m5095,
        "small_mAP@0.5": small,
    }


def run_part3_sahi(
    yw_baseline: dict[str, dict],
    gt_map: dict[str, list[dict]],
    image_paths: list[Path],
    iou_thr: float,
    conf: float = 0.08,
    op_conf: float = 0.2,
    rich_vocab: bool = False,
) -> None:
    """YOLO-World 단독 vs YOLO-World+SAHI 비교. 헤드라인은 mAP/small_mAP."""
    vocab_tag = " [vocab 확장]" if rich_vocab else ""
    print()
    _print_sep()
    print(f"  PART 3: YOLO-World + SAHI 슬라이싱 추론{vocab_tag}")
    print(f"  tile=512×512  overlap=20%  NMS postprocess(IoU={iou_thr})")
    print(f"  conf={conf}  ← 베이스라인 동일 (mAP 공정 비교)")
    _print_sep()

    print("\n[SAHI 추론 중...]")
    sahi_res = _sahi_inference(image_paths, gt_map, conf=conf, iou_thr=iou_thr, rich_vocab=rich_vocab)
    if sahi_res is None:
        return

    # ── 헤드라인: mAP 비교 (conf-독립) ──────────────────────────────────────
    base_m = _summary_metrics(yw_baseline, iou_thr)
    sahi_m = _summary_metrics(sahi_res,    iou_thr)

    def _nan(v): return float("nan") if v is None else v

    print()
    _print_sep("-")
    print("  헤드라인: mAP / small_mAP 전후 비교 (conf=0.08, threshold 통합)")
    _print_sep("-")
    print(f"  {'':28} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'small_mAP@0.5':>14}")
    print("  " + "-" * 66)
    print(f"  {'YW baseline':28} {base_m['mAP@0.5']:>10.4f}"
          f" {base_m['mAP@0.5:0.95']:>14.4f} {_nan(base_m['small_mAP@0.5']):>14.4f}")
    print(f"  {'YW + SAHI':28} {sahi_m['mAP@0.5']:>10.4f}"
          f" {sahi_m['mAP@0.5:0.95']:>14.4f} {_nan(sahi_m['small_mAP@0.5']):>14.4f}")
    print("  " + "-" * 66)
    d_map   = sahi_m["mAP@0.5"]         - base_m["mAP@0.5"]
    d_m95   = sahi_m["mAP@0.5:0.95"]    - base_m["mAP@0.5:0.95"]
    d_small = _nan(sahi_m["small_mAP@0.5"]) - _nan(base_m["small_mAP@0.5"])
    print(f"  {'변화':28} {d_map:>+10.4f} {d_m95:>+14.4f} {d_small:>+14.4f}")
    _print_sep("-")

    # ── 운영점 비교 (conf≥op_conf 필터) ─────────────────────────────────────
    print()
    _print_sep("-")
    print(f"  운영점 비교 (conf≥{op_conf} 필터  /  mAP는 위 표 값 동일)")
    _print_sep("-")
    base_op = _summary_metrics(yw_baseline, iou_thr, conf_filter=op_conf)
    sahi_op = _summary_metrics(sahi_res,    iou_thr, conf_filter=op_conf)
    print(f"  {'':28} {'Recall':>8} {'Precision':>10} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 67)
    print(f"  {'YW baseline (conf≥' + str(op_conf) + ')':28}"
          f" {base_op['recall']:>7.1%} {base_op['precision']:>10.1%}"
          f" {base_op['f1']:>7.1%} {base_op['tp']:>5} {base_op['fp']:>5} {base_op['fn']:>5}")
    print(f"  {'YW + SAHI (conf≥' + str(op_conf) + ')':28}"
          f" {sahi_op['recall']:>7.1%} {sahi_op['precision']:>10.1%}"
          f" {sahi_op['f1']:>7.1%} {sahi_op['tp']:>5} {sahi_op['fp']:>5} {sahi_op['fn']:>5}")
    print("  " + "-" * 67)
    print(f"  {'변화':28}"
          f" {sahi_op['recall']-base_op['recall']:>+7.1%}"
          f" {sahi_op['precision']-base_op['precision']:>+10.1%}"
          f" {sahi_op['f1']-base_op['f1']:>+7.1%}")
    _print_sep("-")

    # ── 이미지별 SAHI 탐지 수 vs 베이스라인 ──────────────────────────────────
    print()
    _print_sep("-")
    print("  이미지별 상세 (SAHI 효과)")
    _print_sep("-")
    print(f"  {'이미지':<42} {'GT':>4} {'YW':>5} {'SAHI':>5}  YW_rec  SA_rec")
    print("  " + "-" * 68)
    for stem in sorted(sahi_res.keys()):
        gt_objs  = sahi_res[stem]["gt"]
        if not gt_objs:
            continue
        yw_preds   = yw_baseline[stem]["preds"]
        sahi_preds = sahi_res[stem]["preds"]
        yw_tp,  _, _ = _greedy_match(yw_preds,   gt_objs, iou_thr)
        sa_tp,  _, _ = _greedy_match(sahi_preds, gt_objs, iou_thr)
        n = len(gt_objs)
        diff = sa_tp - yw_tp
        marker = f"  +{diff}" if diff > 0 else (f"  {diff}" if diff < 0 else "")
        print(f"  {stem[:40]:<40}  {n:>4} {len(yw_preds):>5} {len(sahi_preds):>5}"
              f"  {yw_tp/n:>6.1%}  {sa_tp/n:>6.1%}{marker}")
    _print_sep("-")

    # ── 결론 ─────────────────────────────────────────────────────────────────
    print()
    if d_map > 0.005:
        print(f"  >> SAHI 유효: mAP@0.5 {d_map:+.4f}  small_mAP {d_small:+.4f}")
        print(f"     → base detector: YOLO-World + SAHI 권장")
    elif d_small > 0.005:
        print(f"  >> SAHI 소형 효과: small_mAP {d_small:+.4f}  (전체 mAP {d_map:+.4f})")
        print(f"     → 소형 객체 비중이 높다면 SAHI 적용 권장")
    else:
        print(f"  >> SAHI 효과 미미: mAP {d_map:+.4f}  small_mAP {d_small:+.4f}")
        print(f"     → YOLO-World 단독 유지 (추론 속도 대비 이득 없음)")


# ─── PART 4: vocab conf 스윕 ─────────────────────────────────────────────────

def _print_vocab_conf_sweep(
    baseline_res: dict[str, dict],
    vocab_res: dict[str, dict],
    iou_thr: float,
    conf_levels: tuple[float, ...] = (0.08, 0.15, 0.20, 0.25, 0.30),
) -> None:
    """
    vocab 변형의 conf threshold를 여러 값으로 스윕해 recall/precision/F1 표시.
    baseline conf=0.08 운영점과 나란히 비교 → keeper 판정 근거.
    """
    print()
    _print_sep()
    print("  PART 4: vocab 확장 conf 스윕 (keeper 판정)")
    print("  yolo-world-vocab 에서 conf 임계값을 올려 FP를 줄여도 recall이 유지되는지 확인")
    _print_sep()

    base_op = _summary_metrics(baseline_res, iou_thr, conf_filter=0.08)

    print()
    _print_sep("-")
    hdr = f"  {'설정':32} {'Recall':>8} {'Prec':>8} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(hdr)
    print("  " + "-" * 75)
    print(f"  {'[baseline] yolo-world  conf≥0.08':32}"
          f" {base_op['recall']:>7.1%} {base_op['precision']:>8.1%}"
          f" {base_op['f1']:>7.1%} {base_op['tp']:>5} {base_op['fp']:>5} {base_op['fn']:>5}")
    print("  " + "-" * 75)

    best_conf = None
    best_f1   = -1.0
    for c in conf_levels:
        m = _summary_metrics(vocab_res, iou_thr, conf_filter=c)
        marker = " ◀" if m["f1"] > base_op["f1"] else ""
        print(f"  {'[vocab]  yolo-world-vocab conf≥' + f'{c:.2f}':32}"
              f" {m['recall']:>7.1%} {m['precision']:>8.1%}"
              f" {m['f1']:>7.1%} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}{marker}")
        if m["f1"] > best_f1:
            best_f1   = m["f1"]
            best_conf = c

    _print_sep("-")

    # 판정
    best_m        = _summary_metrics(vocab_res, iou_thr, conf_filter=best_conf)
    recall_gain   = best_m["recall"]    - base_op["recall"]
    f1_gain       = best_m["f1"]        - base_op["f1"]

    if f1_gain > 0.01 and recall_gain >= -0.02:
        verdict = f"✔  vocab keeper: conf≥{best_conf:.2f} 에서 F1 {f1_gain:+.1%}, recall {recall_gain:+.1%}"
    elif recall_gain > 0.05:
        verdict = (f"△  recall 우선이라면 vocab 유리: recall {recall_gain:+.1%}"
                   f" (F1 {f1_gain:+.1%}) — conf≥{best_conf:.2f}")
    else:
        verdict = "✘  vocab: recall/F1 모두 baseline 미달 → baseline 유지"

    print(f"  >> {verdict}")
    _print_sep("-")


# ─── 탐지기 팩토리 ────────────────────────────────────────────────────────────

def _make_detector_factories(names: list[str], conf: float) -> dict:
    """
    탐지기를 즉시 로드하지 않고 callable factory를 반환.
    run_part1이 순차적으로 로드→추론→해제해 MPS 메모리 충돌 방지.
    """
    factories: dict = {}
    for name in names:
        if name == "florence-drc":
            factories[name] = lambda: FlorenceDRCDetector()
        else:
            factories[name] = lambda n=name: get_detector(n, conf=conf)
    return factories


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot detector 평가 (YOLO-World / YOLOE / Florence-DRC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-dir", type=Path, default=_DEFAULT_TEST_DIR,
        help=f"테스트 이미지 디렉터리 (기본: {_DEFAULT_TEST_DIR})",
    )
    parser.add_argument(
        "--detectors", nargs="+",
        default=["yolo-world", "yoloe", "florence-drc"],
        choices=["yolo-world", "yoloe", "florence-drc", "yolo-world-vocab"],
        help="실행할 탐지기 목록 (기본: 셋 모두)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.08,
        help="YOLO 계열 confidence threshold (기본: 0.08)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="TP 판정 IoU threshold (기본: 0.5)",
    )
    args = parser.parse_args()

    test_dir: Path = args.test_dir
    coco_json = test_dir / "_annotations.coco.json"

    if not test_dir.exists():
        print(f"[ERROR] 테스트 디렉터리 없음: {test_dir}")
        print("  --test-dir 옵션으로 경로를 직접 지정하세요.")
        sys.exit(1)
    if not coco_json.exists():
        print(f"[ERROR] COCO annotations 없음: {coco_json}")
        sys.exit(1)

    # GT 로드
    gt_map = load_coco_gt(coco_json)
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    total_gt = sum(len(v) for v in gt_map.values())

    print()
    _print_sep()
    print(f"  도메인 Zero-shot Detector 평가")
    print(f"  테스트 이미지: {len(image_paths)}장")
    print(f"  GT 객체 수:    {total_gt}개")
    print(f"  탐지기:        {', '.join(args.detectors)}")
    print(f"  conf={args.conf}  iou={args.iou}")
    _print_sep()

    # 탐지기 팩토리 준비 (실제 로드는 run_part1 내부에서 순차 수행)
    factories = _make_detector_factories(args.detectors, conf=args.conf)
    if not factories:
        print("[ERROR] 사용 가능한 탐지기 없음")
        sys.exit(1)

    # PART 1
    print()
    all_results = run_part1(factories, image_paths, gt_map, iou_thr=args.iou)

    # PART 2
    run_part2(all_results, iou_thr=args.iou)

    # PART 3: SAHI — 항상 yolo-world 순수 베이스라인 기준으로 비교
    if "yolo-world" in all_results:
        run_part3_sahi(
            yw_baseline=all_results["yolo-world"],
            gt_map=gt_map,
            image_paths=image_paths,
            iou_thr=args.iou,
            conf=args.conf,
            op_conf=0.2,
            rich_vocab=False,
        )

    # PART 4: vocab conf 스윕 (keeper 판정)
    if "yolo-world" in all_results and "yolo-world-vocab" in all_results:
        _print_vocab_conf_sweep(
            baseline_res=all_results["yolo-world"],
            vocab_res=all_results["yolo-world-vocab"],
            iou_thr=args.iou,
        )

    print()
    _print_sep()
    print("  평가 완료")
    _print_sep()


if __name__ == "__main__":
    main()
