import os
import torch


MODEL_PATH      = os.getenv("MODEL_PATH",      "yolov8s-world.pt")
DETECTOR_NAME   = os.getenv("DETECTOR",        "yolo-world")
CONF_THRESHOLD  = float(os.getenv("CONF",      "0.08"))
IMGSZ           = int(os.getenv("IMGSZ",       "960"))
STATIC_DIR      = os.getenv("STATIC_DIR",      "outputs/static")
CLAUDE_MODEL    = os.getenv("CLAUDE_MODEL",    "claude-haiku-4-5-20251001")


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
