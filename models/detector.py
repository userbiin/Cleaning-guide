"""
Detector wrappers — all return list[dict] with {label, confidence, bbox:[x1,y1,x2,y2]}.
Device priority: cuda > mps > cpu  (auto-detected unless overridden)
"""

import torch

OBJECT_CLASSES = [
    # 문구/학습
    "book", "notebook", "paper", "document", "pen", "pencil",
    "marker", "scissors", "ruler", "eraser",
    # 전자기기
    "laptop", "keyboard", "mouse", "monitor", "phone",
    "tablet", "charger", "cable", "earphones", "remote control",
    # 식기/음식
    "cup", "bottle", "mug", "plate", "bowl", "spoon",
    "fork", "snack", "food wrapper",
    # 의류/잡화
    "clothes", "shirt", "hoodie", "pants", "socks",
    "bag", "backpack", "hat", "shoe",
    # 화장품/생활용품
    "cosmetics", "tissue", "medicine", "toothbrush", "comb", "mirror",
    # 쓰레기/기타
    "trash", "box", "plastic bag", "toy",
]


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YOLOWorldDetector:
    """YOLO-World open-vocabulary detector (yolov8s-world.pt)."""

    def __init__(self, model_path: str = "yolov8s-world.pt",
                 conf: float = 0.08, imgsz: int = 960, device: str | None = None):
        from ultralytics import YOLO
        self.device = device or _auto_device()
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)
        self.model.set_classes(OBJECT_CLASSES + [""])  # "" = background 버퍼

    def detect(self, image_path: str) -> list[dict]:
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        return self._parse(results[0])

    def _parse(self, result) -> list[dict]:
        if result.boxes is None:
            return []
        out = []
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            if label == "":
                continue
            out.append({
                "label": label,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": list(map(int, box.xyxy[0])),
            })
        return out


class YOLOEDetector:
    """YOLOE open-vocabulary detector (requires yoloe-11s.pt or similar).
    모델 파일이 없으면 ultralytics가 자동 다운로드합니다.
    """

    def __init__(self, model_path: str = "yoloe-11s.pt",
                 conf: float = 0.08, imgsz: int = 960, device: str | None = None):
        from ultralytics import YOLO
        self.device = device or _auto_device()
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)
        # YOLOE open-vocab: set_classes with text embeddings
        try:
            pe = self.model.get_text_pe(OBJECT_CLASSES)
            self.model.set_classes(OBJECT_CLASSES, pe)
        except AttributeError:
            # ultralytics 버전에 따라 get_text_pe 없을 수 있음
            self.model.set_classes(OBJECT_CLASSES)

    def detect(self, image_path: str) -> list[dict]:
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        return self._parse(results[0])

    def _parse(self, result) -> list[dict]:
        if result.boxes is None:
            return []
        out = []
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            out.append({
                "label": label,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": list(map(int, box.xyxy[0])),
            })
        return out


_REGISTRY: dict[str, type] = {
    "yolo-world": YOLOWorldDetector,
    "yoloe": YOLOEDetector,
}


def get_detector(name: str, **kwargs):
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown detector '{name}'. Available: {list(_REGISTRY)}")
    return cls(**kwargs)
