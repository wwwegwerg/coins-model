from ultralytics import YOLO

from .config import DATA_YAML, IMGSZ


def validate_best_model(best_path: str, run_name: str) -> YOLO:
    """Валидирует модель и сохраняет результаты в runs/val/<run_name>/."""
    best_model = YOLO(best_path)
    metrics = best_model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        project="runs/val",
        name=run_name,
    )

    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")

    return best_model
