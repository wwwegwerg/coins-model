from datetime import datetime
from pathlib import Path

from .config import BATCH, DATA_YAML, DEVICE, IMGSZ, WORKERS


def make_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


COMMON_TRAIN_ARGS = dict(
    data=DATA_YAML,
    device=DEVICE,
    amp=True,
    workers=WORKERS,
    imgsz=IMGSZ,
    batch=BATCH,
    optimizer="auto",
    patience=20,
    save_period=10,
    cache=False,
    # аугментации
    # auto_augment=None,
    # erasing=0.0,
    # mosaic=0.0,
)


def validate_mode(mode: str) -> None:
    if mode not in {"train", "resume", "finetune"}:
        raise ValueError("MODE должен быть 'train', 'resume' или 'finetune'")


def extract_run_name(save_dir: str) -> str:
    """Извлекает имя run из пути save_dir.
    
    Пример: 'runs/train/train_20260312_040919' -> 'train_20260312_040919'
    Работает кроссплатформенно (Windows/Linux/macOS).
    """
    return Path(save_dir).name
