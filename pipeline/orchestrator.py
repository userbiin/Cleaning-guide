"""
추론 파이프라인: 전처리 → 탐지 → (describe) → attribute → category → LLM.
describe=False  → OFF 경로 (대조군, P1)
describe=True   → ON 경로  (처치군, P3 이후)
"""

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # pillow-heif 미설치 시 HEIC 지원 비활성화

import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

from pipeline.category import classify_item
from pipeline.attributes import parse_state_signal
from pipeline.llm import build_prompt, ask_llm

STATIC_DIR = Path(os.getenv("STATIC_DIR", "outputs/static"))
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ── 전처리 ─────────────────────────────────────────────────────────────────────

def _load_image_corrected(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """EXIF 회전 보정 후 BGR ndarray 반환. (orig_w, orig_h)도 함께."""
    pil = Image.open(image_path)
    orig_w, orig_h = pil.size

    # EXIF 회전 보정
    try:
        exif = pil._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if val == 3:   pil = pil.rotate(180, expand=True)
                    elif val == 6: pil = pil.rotate(270, expand=True)
                    elif val == 8: pil = pil.rotate(90,  expand=True)
    except Exception:
        pass

    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    return bgr, (orig_w, orig_h)


# ── 어노테이션 저장 ─────────────────────────────────────────────────────────────

_DECISION_COLORS = {
    "discard":  (0,   0,   220),   # 빨강
    "relocate": (0,  165,  255),   # 주황
    "keep":     (0,  200,   80),   # 초록
    None:       (180, 180, 180),   # 회색 (LLM 미호출)
}


def _annotate_and_save(bgr: np.ndarray, objects: list[dict], image_id: str) -> str:
    """어노테이션 이미지를 STATIC_DIR에 저장하고 URL path 반환."""
    img = bgr.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        color = _DECISION_COLORS.get(obj.get("decision"), _DECISION_COLORS[None])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"[{obj.get('decision','?')}] {obj['label']} {obj['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    filename = f"{image_id}_detected.jpg"
    cv2.imwrite(str(STATIC_DIR / filename), img)
    return f"/static/{filename}"


# ── 메인 파이프라인 ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    detector,              # 사전 로드된 detector 인스턴스
    describer=None,        # 사전 로드된 Florence2Describer (describe=True 시 필요)
    describe: bool = False,
) -> dict:
    t0 = time.time()
    image_id = str(uuid.uuid4())[:8]

    # 1. 전처리
    bgr, _ = _load_image_corrected(image_path)
    proc_h, proc_w = bgr.shape[:2]

    # 전처리된 이미지를 임시 저장 (detector는 파일 경로 입력)
    tmp_path = str(STATIC_DIR / f"{image_id}_tmp.jpg")
    cv2.imwrite(tmp_path, bgr)

    # 2. 탐지
    raw_preds = detector.detect(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    # 3. 객체별 처리
    objects = []
    for i, pred in enumerate(raw_preds):
        cat_key, cat_ko, _ = classify_item(pred["label"])

        description: str | None = None
        state: str | None = None

        # describe=ON 경로 (P3 이후 — describer가 주입된 경우만)
        if describe and describer is not None:
            description = describer.describe_region(
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                pred["bbox"],
            )
            state = parse_state_signal(description)

        objects.append({
            "id":          i,
            "label":       pred["label"],
            "description": description,
            "state":       state,
            "category":    cat_key,
            "category_ko": cat_ko,
            "decision":    None,         # LLM이 채움
            "confidence":  pred["confidence"],
            "bbox":        pred["bbox"],
        })

    # 4. LLM 판단
    summary = ""
    guide: dict = {}

    if objects:
        prompt = build_prompt(objects, describe=describe)
        llm_result = ask_llm(prompt)

        if llm_result:
            # decision 주입
            for dec in llm_result.get("decisions", []):
                oid = dec.get("object_id")
                if oid is not None and 0 <= oid < len(objects):
                    objects[oid]["decision"] = dec.get("decision")
            guide = llm_result.get("guide", {})

            # 요약 문장 생성
            discard_cnt  = sum(1 for o in objects if o["decision"] == "discard")
            relocate_cnt = sum(1 for o in objects if o["decision"] == "relocate")
            summary = (f"총 {len(objects)}개 물건 탐지. "
                       f"버릴 것 {discard_cnt}개, 옮길 것 {relocate_cnt}개.")
        else:
            summary = f"총 {len(objects)}개 물건 탐지 (LLM 미호출 — OPENAI_API_KEY 확인)."

    # 5. 어노테이션 저장
    annotated_url = _annotate_and_save(bgr, objects, image_id)

    return {
        "image_id":            image_id,
        "processed_image_size": [proc_w, proc_h],
        "summary":             summary,
        "objects":             objects,
        "guide":               guide,
        "annotated_image_url": annotated_url,
        "latency_ms":          int((time.time() - t0) * 1000),
    }
