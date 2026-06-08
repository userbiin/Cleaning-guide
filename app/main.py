"""
FastAPI 백엔드 — POST /analyze

실행:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

환경변수:
  OPENAI_API_KEY   LLM 호출 (없으면 탐지·카테고리까지만 동작)
  MODEL_PATH       yolov8s-world.pt (기본)
  DETECTOR         yolo-world | yoloe (기본: yolo-world)
  CONF             탐지 threshold (기본: 0.08)
  STATIC_DIR       어노테이션 이미지 저장 경로 (기본: outputs/static)
"""

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    CONF_THRESHOLD, DETECTOR_NAME, IMGSZ,
    MODEL_PATH, STATIC_DIR, auto_device,
)
from app.schemas import AnalyzeResponse, DetectedObject, Guide
from models.detector import get_detector
from pipeline.orchestrator import run_pipeline

STATIC_PATH = Path(STATIC_DIR)
STATIC_PATH.mkdir(parents=True, exist_ok=True)


# ── lifespan: 서버 기동 시 모델 사전 로드 ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] detector={DETECTOR_NAME}  device={auto_device()}  conf={CONF_THRESHOLD}")
    app.state.detector = get_detector(
        DETECTOR_NAME,
        model_path=MODEL_PATH,
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        device=auto_device(),
    )
    app.state.describer = None  # P3에서 Florence2Describer 주입 예정
    print("[startup] 모델 로드 완료")
    yield
    print("[shutdown] 종료")


app = FastAPI(title="Organization Detector API", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


# ── /health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    import os
    return {
        "status": "ok",
        "detector": DETECTOR_NAME,
        "device": auto_device(),
        "llm": "enabled" if os.getenv("OPENAI_API_KEY") else "disabled (no API key)",
    }


# ── POST /analyze ──────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    describe: bool    = Form(False),
    detector: str     = Form("yolo-world"),
):
    # 지원 형식 검사
    allowed = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=422,
                            detail=f"지원 형식: {allowed}. 받은 파일: {image.filename}")

    # 업로드 이미지를 임시 파일에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_pipeline(
            image_path=tmp_path,
            detector=request.app.state.detector,
            describer=request.app.state.describer,
            describe=describe,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Guide 파싱 (LLM이 dict 반환, 없으면 빈 Guide)
    raw_guide = result.get("guide", {})
    guide = Guide(
        priority_top3=raw_guide.get("priority_top3", []),
        five_min_routine=raw_guide.get("five_min_routine", []),
        by_category=raw_guide.get("by_category", {}),
    )

    objects = [DetectedObject(**obj) for obj in result["objects"]]

    return AnalyzeResponse(
        image_id=result["image_id"],
        processed_image_size=result["processed_image_size"],
        summary=result["summary"],
        objects=objects,
        guide=guide,
        annotated_image_url=result["annotated_image_url"],
        latency_ms=result["latency_ms"],
    )


# ── 예외 핸들러 ────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})
