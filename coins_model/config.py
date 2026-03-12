from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


PATHS = {
    "runs_root": PROJECT_ROOT / "runs",
    "train_dir": PROJECT_ROOT / "runs" / "train",
    "val_dir": PROJECT_ROOT / "runs" / "val",
    "export_dir": PROJECT_ROOT / "runs" / "export",
}


RUN = {
    # Режимы:
    # "train"    -> новая тренировка с нуля от pretrained модели
    # "resume"   -> продолжить незавершенный run через resume=True
    # "finetune" -> дообучение завершенного run как новый запуск от checkpoint
    "mode": "train",
    # Можно выбрать: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
    "model_name": "yolo26n.pt",
    # Путь к checkpoint для resume/finetune.
    # Должен указывать на runs/train/<run_name>/weights/last.pt или best.pt
    "checkpoint_path": "",
    # Для train и finetune это число эпох текущего запуска
    "epochs": 10,
    # Для resume это ОБЩЕЕ число эпох после продолжения.
    # Например, если было 10 и хочешь еще 10, ставь 20.
    "resume_total_epochs": 20,
}


RUN_PREFIXES = {
    "train": "train",
    "finetune": "finetune",
}


TRAIN = {
    "data_yaml": PROJECT_ROOT / "dataset_coins" / "data.yaml",
    "device": "mps",
    "imgsz": 640,
    "batch": 12,
    "workers": 2,
    "amp": True,
    "optimizer": "auto",
    "patience": 20,
    "save_period": 10,
    "cache": False,
}


VALIDATE = {
    "data_yaml": PROJECT_ROOT / "dataset_coins" / "data.yaml",
    "imgsz": 640,
}


EXPORT = {
    "format": "onnx",
}
