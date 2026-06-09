"""
Florence-2-base 기반 Dense Captioner — models/describer.py

orchestrator.py의 describe=ON 경로에서 호출:
  from models.describer import Florence2Describer
  describer = Florence2Describer()
  text = describer.describe_region(pil_rgb_image, [x1, y1, x2, y2])
"""

from __future__ import annotations

import torch
from PIL import Image

MODEL_ID = "microsoft/Florence-2-base"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Florence2Describer:
    """
    Florence-2-base 래퍼.
    bbox 영역을 pad-crop한 뒤 <MORE_DETAILED_CAPTION> 프롬프트로 describe.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        max_new_tokens: int = 512,
        num_beams: int = 3,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

        self.device = device or _auto_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.max_new_tokens = max_new_tokens
        # beam search는 CUDA에서만 안정적 — MPS·CPU는 greedy(num_beams=1)
        self.num_beams = num_beams if self.device == "cuda" else 1

        print(f"[Florence-2] {model_id} 로드 중 (device={self.device}, dtype={self.dtype})...")

        # SDPA 가속: CUDA는 float16+SDPA, MPS/CPU는 eager(float32)
        attn_impl = "sdpa" if self.device == "cuda" else "eager"

        # 최신 transformers에서 PretrainedConfig 클래스 속성 forced_bos_token_id가 제거됨.
        # Florence2LanguageConfig.__init__이 super().__init__() 호출 전에 self.forced_bos_token_id에
        # 접근하므로 PretrainedConfig 클래스 레벨에 기본값을 복원해야 함.
        from transformers import PretrainedConfig, AutoConfig
        if not hasattr(PretrainedConfig, "forced_bos_token_id"):
            PretrainedConfig.forced_bos_token_id = None

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                dtype=self.dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            ).eval().to(self.device)
        except Exception:
            # attn_implementation 미지원 시 eager fallback
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                dtype=self.dtype,
                trust_remote_code=True,
            ).eval().to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("[Florence-2] 로드 완료")

    # ── 내부 추론 ────────────────────────────────────────────────────────────

    def _run(self, image: Image.Image, task: str) -> dict:
        """Florence-2 단일 태스크 실행. 결과 dict 반환."""
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        input_ids    = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        return self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def describe_region(
        self,
        image: Image.Image,
        bbox: list[int],
        pad: float = 0.08,
    ) -> str:
        """
        bbox=[x1,y1,x2,y2] 영역을 pad_ratio만큼 여백 확장 후 crop,
        <MORE_DETAILED_CAPTION>으로 상세 설명 생성.
        반환값: 문자열 (빈 bbox이면 "")
        """
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            return ""

        W, H = image.size
        crop = image.crop((
            max(0, int(x1 - bw * pad)),
            max(0, int(y1 - bh * pad)),
            min(W, int(x2 + bw * pad)),
            min(H, int(y2 + bh * pad)),
        )).convert("RGB")

        if crop.width < 8 or crop.height < 8:
            return ""

        result = self._run(crop, "<MORE_DETAILED_CAPTION>")
        return result.get("<MORE_DETAILED_CAPTION>", "").strip()

    def dense_caption(self, image: Image.Image) -> str:
        """전체 이미지 <MORE_DETAILED_CAPTION> (파이프라인 디버깅용)."""
        result = self._run(image.convert("RGB"), "<MORE_DETAILED_CAPTION>")
        return result.get("<MORE_DETAILED_CAPTION>", "").strip()
