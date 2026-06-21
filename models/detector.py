"""
Detector wrappers — all return list[dict] with {label, confidence, bbox:[x1,y1,x2,y2]}.
Device priority: cuda > mps > cpu  (auto-detected unless overridden)
"""

import torch


# detector.py
OBJECT_CLASSES = [
    # 침구
    "blanket",          # 1

    # 문구/학습
    "book",             # 2
    "cd",               # 13
    "document",         # 17
    "eraser",           # 19
    "folder",           # 20
    "memo",             # 29
    "notebook",         # 33
    "paper",            # 34
    "pen",              # 35
    "pencil case",      # 36

    # 전자기기
    "box",              # 6
    "cable",            # 7
    "calculator",       # 8
    "camera",           # 10
    "charger",          # 14
    "earphones",        # 18
    "headset",          # 22
    "keyboard",         # 24
    "lamp",             # 25
    "laptop",           # 26
    "monitor",          # 30
    "mouse",            # 31
    "mouse pad",        # 32
    "phone",            # 37
    "printer",          # 40
    "remote controller",# 41
    "speaker",          # 43

    # 용기/식품
    "can",              # 11
    "cup",              # 15
    "kettle",           # 23
    "snack",            # 42

    # 수납/잡화
    "case",             # 12
    "pouch",            # 39

    # 생활용품
    "hair brush",       # 21
    "mask",             # 27
    "medicine",         # 28
    "sunglasses",       # 44
    "tape",             # 45
    "tissue",           # 46

    # 기타
    "toy",              # 47
    "trash",            # 48
    "calendar",         # 9 
]


# ── 어휘 확장 테이블 ─────────────────────────────────────────────────────────
# GT 라벨(canonical)은 그대로 두고, YOLO-World에 주는 프롬프트만 풍부하게.
# 탐지 결과는 다시 canonical 라벨로 역매핑.
# 비어있는 클래스는 canonical 이름 단독 사용.
PROMPT_VOCAB: dict[str, list[str]] = {
    # 침구
    "blanket": ["blanket", "throw blanket", "bed blanket"],

    # 문구/학습
    "book": ["book", "open book", "textbook", "paperback book"],
    "cd": ["cd", "dvd disc", "optical disc"],
    "document": ["document", "official paper document", "printed document"],
    "eraser": ["eraser", "rubber eraser"],
    "folder": ["folder", "file folder", "document folder", "plastic folder"],
    "memo": ["memo", "sticky note", "post-it note"],
    "notebook": ["notebook", "spiral notebook", "ruled notebook"],
    "paper": [
        "paper", "sheet of paper", "white paper", "printed paper",
        "loose paper", "papers on desk",
    ],
    "pen": ["pen", "ballpoint pen", "felt-tip pen", "ink pen"],
    "pencil case": ["pencil case", "pencil bag", "pen holder case"],

    # 전자기기
    "box": ["box", "cardboard box", "storage box", "packaging box"],
    "cable": [
        "cable", "usb cable", "charging cable",
        "power cord", "black wire", "tangled cable",
    ],
    "calculator": ["calculator", "pocket calculator"],
    "camera": ["camera", "digital camera"],
    "charger": [
        "charger", "phone charger", "usb charger",
        "charging adapter", "power adapter",
    ],
    "earphones": ["earphones", "earbuds", "in-ear headphones"],
    "headset": ["headset", "headphones", "over-ear headphones"],
    "keyboard": ["keyboard", "computer keyboard"],
    "lamp": ["lamp", "desk lamp", "table lamp"],
    "laptop": ["laptop", "laptop computer"],
    "monitor": ["monitor", "computer monitor", "display screen"],
    "mouse": ["mouse", "computer mouse", "wireless mouse"],
    "mouse pad": ["mouse pad", "mousepad", "desk pad"],
    "phone": ["phone", "smartphone", "cell phone", "mobile phone"],
    "printer": ["printer", "inkjet printer"],
    "remote controller": ["remote controller", "remote control", "tv remote"],
    "speaker": ["speaker", "computer speaker", "desktop speaker"],

    # 용기/식품
    "can": ["can", "aluminum can", "soda can"],
    "cup": ["cup", "coffee cup", "mug", "drinking cup", "tumbler"],
    "kettle": ["kettle", "water kettle", "electric kettle"],
    "snack": ["snack", "snack bag", "chips bag", "snack package"],

    # 수납/잡화
    "case": ["case", "storage case", "plastic case", "hard case"],
    "pouch": ["pouch", "zipper pouch", "small pouch", "fabric pouch"],

    # 생활용품
    "hair brush": ["hair brush", "hairbrush"],
    "mask": ["mask", "face mask", "surgical mask", "disposable mask"],
    "medicine": ["medicine", "pill bottle", "medication bottle"],
    "sunglasses": ["sunglasses", "eyeglasses", "glasses"],
    "tape": ["tape", "adhesive tape", "scotch tape", "tape roll"],
    "tissue": ["tissue", "tissue box", "facial tissue"],

    # 기타
    "toy": ["toy", "figurine", "plush toy"],
    "trash": ["trash", "garbage", "waste", "empty packaging"],
    "calendar": ["calendar", "desk calendar", "wall calendar"],
}


def build_prompt_list() -> tuple[list[str], dict[str, str]]:
    """
    PROMPT_VOCAB → (flat_prompts, canonical_map).
    flat_prompts : YOLO-World set_classes()에 넘길 프롬프트 리스트.
    canonical_map: {prompt → canonical OBJECT_CLASS 이름}
    """
    prompts: list[str] = []
    canonical_map: dict[str, str] = {}
    seen: set[str] = set()

    for canonical in OBJECT_CLASSES:
        variants = PROMPT_VOCAB.get(canonical, [])
        # canonical 자체가 없으면 맨 앞에 삽입
        if canonical not in variants:
            variants = [canonical] + list(variants)
        for p in variants:
            if p not in seen:
                prompts.append(p)
                canonical_map[p] = canonical
                seen.add(p)

    return prompts, canonical_map


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YOLOWorldDetector:
    """YOLO-World open-vocabulary detector (yolov8s-world.pt).

    rich_vocab=True  : PROMPT_VOCAB의 확장 프롬프트로 set_classes,
                       예측 라벨을 canonical OBJECT_CLASS로 역매핑.
    rich_vocab=False : 기존 OBJECT_CLASSES 단어 그대로 사용(베이스라인).
    """

    def __init__(self, model_path: str = "yolov8s-world.pt",
                 conf: float = 0.08, imgsz: int = 1280,
                 device: str | None = None, rich_vocab: bool = False):
        from ultralytics import YOLO
        self.device = device or _auto_device()
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)

        if rich_vocab:
            prompts, self._canonical_map = build_prompt_list()
            self._classes = prompts + [""]
        else:
            self._canonical_map: dict[str, str] = {}
            self._classes = OBJECT_CLASSES + [""]
        self.model.set_classes(self._classes)

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
            raw_label = result.names[int(box.cls[0])]
            if raw_label == "":
                continue
            label = self._canonical_map.get(raw_label, raw_label)
            out.append({
                "label": label,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": list(map(int, box.xyxy[0])),
            })
        return out


class YOLOEDetector:
    """YOLOE open-vocabulary detector.
    yoloe-11s-seg.pt 는 ultralytics v8.3+ 에서 자동 다운로드됩니다.
    """

    def __init__(self, model_path: str = "yoloe-11s-seg.pt",
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
    "yolo-world":       YOLOWorldDetector,
    "yoloe":            YOLOEDetector,
}


def get_detector(name: str, **kwargs):
    if name == "yolo-world-vocab":
        kwargs.setdefault("rich_vocab", True)
        return YOLOWorldDetector(**kwargs)
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{name}'. "
            f"Available: {list(_REGISTRY) + ['yolo-world-vocab']}"
        )
    return cls(**kwargs)
