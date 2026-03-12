from ultralytics import YOLO

from ..config import EPOCHS, MODEL_NAME
from ..shared import COMMON_TRAIN_ARGS, extract_run_name, make_run_name


def run_train():
    model = YOLO(MODEL_NAME)
    run_name = make_run_name("train")

    results = model.train(
        **COMMON_TRAIN_ARGS,
        epochs=EPOCHS,
        project="runs/train",
        name=run_name,
    )

    print("Новая тренировка завершена.")
    print("Результаты сохранены в:", results.save_dir)
    return results, run_name
