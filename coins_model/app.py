import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from .config import EXPORT, PATHS, RUN, RUN_PREFIXES, TRAIN, VALIDATE


def make_run_name(mode: str) -> str:
    prefix = RUN_PREFIXES[mode]
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def extract_run_name(save_dir: str) -> str:
    return Path(save_dir).name


def validate_mode(mode: str) -> None:
    if mode not in {"train", "resume", "finetune"}:
        raise ValueError("RUN['mode'] должен быть 'train', 'resume' или 'finetune'")


def resolve_checkpoint_path() -> Path:
    checkpoint_path = RUN["checkpoint_path"]
    if not checkpoint_path:
        raise ValueError("Для режима resume/finetune нужно заполнить RUN['checkpoint_path']")

    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = (PATHS["runs_root"].parent / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {path}")

    return path


def ensure_output_dirs() -> None:
    PATHS["train_dir"].mkdir(parents=True, exist_ok=True)
    PATHS["val_dir"].mkdir(parents=True, exist_ok=True)
    PATHS["export_dir"].mkdir(parents=True, exist_ok=True)


def build_train_kwargs() -> dict:
    return {
        "data": str(TRAIN["data_yaml"]),
        "device": TRAIN["device"],
        "amp": TRAIN["amp"],
        "workers": TRAIN["workers"],
        "imgsz": TRAIN["imgsz"],
        "batch": TRAIN["batch"],
        "optimizer": TRAIN["optimizer"],
        "patience": TRAIN["patience"],
        "save_period": TRAIN["save_period"],
        "cache": TRAIN["cache"],
    }


def run_train():
    model = YOLO(RUN["model_name"])
    run_name = make_run_name("train")

    results = model.train(
        **build_train_kwargs(),
        epochs=RUN["epochs"],
        project=str(PATHS["train_dir"]),
        name=run_name,
    )

    print("Новая тренировка завершена.")
    print("Результаты сохранены в:", results.save_dir)
    return results, run_name


def run_resume():
    checkpoint_path = resolve_checkpoint_path()
    model = YOLO(str(checkpoint_path))

    results = model.train(
        resume=True,
        epochs=RUN["resume_total_epochs"],
    )

    print("Незавершенный run продолжен.")
    print("Результаты сохранены в:", results.save_dir)
    run_name = extract_run_name(results.save_dir)
    return results, run_name


def run_finetune():
    checkpoint_path = resolve_checkpoint_path()
    model = YOLO(str(checkpoint_path))
    run_name = make_run_name("finetune")

    results = model.train(
        **build_train_kwargs(),
        epochs=RUN["epochs"],
        project=str(PATHS["train_dir"]),
        name=run_name,
    )

    print("Дообучение завершено.")
    print("Результаты сохранены в:", results.save_dir)
    return results, run_name


def validate_best_model(best_path: Path, run_name: str) -> YOLO:
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(VALIDATE["data_yaml"]),
        imgsz=VALIDATE["imgsz"],
        project=str(PATHS["val_dir"]),
        name=run_name,
    )

    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")

    return best_model


def export_model(best_model: YOLO, run_name: str) -> Path:
    export_dir = PATHS["export_dir"] / run_name
    export_dir.mkdir(parents=True, exist_ok=True)

    exported_path = Path(best_model.export(format=EXPORT["format"]))
    final_path = export_dir / exported_path.name

    if final_path.exists():
        final_path.unlink()

    shutil.move(str(exported_path), str(final_path))
    print(f"Экспорт завершён: {final_path}")
    return final_path


def main():
    validate_mode(RUN["mode"])
    ensure_output_dirs()

    if RUN["mode"] == "train":
        results, run_name = run_train()
    elif RUN["mode"] == "resume":
        results, run_name = run_resume()
    else:
        results, run_name = run_finetune()

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = validate_best_model(best_path, run_name)
    export_model(best_model, run_name)
