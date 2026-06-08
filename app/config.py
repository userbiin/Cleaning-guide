import os
import torch


MODEL_PATH      = os.getenv("MODEL_PATH",      "yolov8s-world.pt")
DETECTOR_NAME   = os.getenv("DETECTOR",        "yolo-world")
CONF_THRESHOLD  = float(os.getenv("CONF",      "0.08"))
IMGSZ           = int(os.getenv("IMGSZ",       "960"))
STATIC_DIR      = os.getenv("STATIC_DIR",      "outputs/static")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o-mini")


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
