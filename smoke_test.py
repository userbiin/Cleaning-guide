"""
백엔드 smoke test — FastAPI 없이 파이프라인 직접 실행.
detector + describer(describe=True) 전부 사용.

실행:
    python smoke_test.py                        # test_images/ 전체
    python smoke_test.py test_images/test1.jpeg # 특정 파일
"""

import sys
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from app.config import CONF_THRESHOLD, DETECTOR_NAME, IMGSZ, MODEL_PATH, auto_device
from models.detector import get_detector
from models.describer import Florence2Describer
from pipeline.orchestrator import run_pipeline

# ── 이미지 목록 ────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    images = sys.argv[1:]
else:
    images = sorted(glob.glob("test_images/*.jpeg") + glob.glob("test_images/*.jpg") +
                    glob.glob("test_images/*.png"))

if not images:
    print("test_images/ 에 이미지가 없습니다.")
    sys.exit(1)

# ── 모델 로드 (최초 1회) ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"device={auto_device()}  detector={DETECTOR_NAME}  conf={CONF_THRESHOLD}")
print(f"{'='*60}\n")

detector = get_detector(DETECTOR_NAME, model_path=MODEL_PATH,
                        conf=CONF_THRESHOLD, imgsz=IMGSZ, device=auto_device())

try:
    describer = Florence2Describer()
except Exception as e:
    print(f"[경고] Florence-2 로드 실패 — describe=OFF로 진행: {e}\n")
    describer = None

# ── 이미지별 실행 ──────────────────────────────────────────────────────────────
for img_path in images:
    if not Path(img_path).exists():
        print(f"[건너뜀] 파일 없음: {img_path}")
        continue

    print(f"\n{'─'*60}")
    print(f"이미지: {img_path}")
    print(f"{'─'*60}")

    result = run_pipeline(
        image_path=img_path,
        detector=detector,
        describer=describer,
        describe=(describer is not None),
    )

    print(f"요약  : {result['summary']}")
    print(f"크기  : {result['processed_image_size']}  |  처리시간: {result['latency_ms']} ms")
    print(f"어노테이션: {result['annotated_image_url']}")

    objects = result["objects"]
    if not objects:
        print("탐지된 물건 없음.")
        continue

    print(f"\n탐지된 물건 ({len(objects)}개):")
    for obj in objects:
        decision = obj.get("decision") or "미정"
        desc     = obj.get("description") or ""
        state    = obj.get("state") or ""
        print(f"  [{obj['id']:2d}] {obj['label']:<20s} "
              f"conf={obj['confidence']:.2f}  "
              f"cat={obj['category_ko']:<12s}  "
              f"→ {decision}")
        if desc:
            print(f"        설명: {desc}")
        if state:
            print(f"        상태: {state}")

    guide = result.get("guide", {})
    if guide.get("priority_top3"):
        print("\n정리 우선순위:")
        for i, item in enumerate(guide["priority_top3"], 1):
            print(f"  {i}. {item}")
    if guide.get("five_min_routine"):
        print("\n5분 루틴:")
        for i, step in enumerate(guide["five_min_routine"], 1):
            print(f"  {i}. {step}")

print(f"\n{'='*60}")
print("완료.")
