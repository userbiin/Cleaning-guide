from pydantic import BaseModel


class DetectedObject(BaseModel):
    id:          int
    label:       str
    description: str | None = None   # describe=ON 시 채워짐
    state:       str | None = None   # describe=ON 시 채워짐
    category:    str
    category_ko: str
    decision:    str | None = None   # LLM 판정
    confidence:  float
    bbox:        list[int]           # [x1, y1, x2, y2]


class Guide(BaseModel):
    priority_top3:   list[str] = []
    five_min_routine: list[str] = []
    by_category:     dict[str, str] = {}


class AnalyzeResponse(BaseModel):
    image_id:             str
    processed_image_size: list[int]   # [w, h] — iOS bbox 스케일링용
    summary:              str
    objects:              list[DetectedObject]
    guide:                Guide
    annotated_image_url:  str
    latency_ms:           int
