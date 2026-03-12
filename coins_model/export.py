import os
from pathlib import Path

from ultralytics import YOLO


def export_onnx(model: YOLO, run_name: str) -> None:
    """Экспортирует модель в ONNX и сохраняет в runs/export/<run_name>/."""
    export_dir = Path("runs/export") / run_name
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Ultralytics export сохраняет рядом с моделью, поэтому нужно временно
    # изменить рабочую директорию или использовать явный путь
    original_cwd = os.getcwd()
    try:
        os.chdir(export_dir)
        model.export(format="onnx")
        print(f"Экспорт в ONNX завершён: {export_dir}")
    finally:
        os.chdir(original_cwd)
