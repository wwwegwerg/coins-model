from ultralytics import YOLO

from ..config import CKPT_PATH, EPOCHS
from ..shared import COMMON_TRAIN_ARGS, make_run_name


def run_finetune():
    model = YOLO(CKPT_PATH)
    run_name = make_run_name("finetune")

    results = model.train(
        **COMMON_TRAIN_ARGS,
        epochs=EPOCHS,
        project="runs/train",
        name=run_name,
    )

    print("Дообучение завершено.")
    print("Результаты сохранены в:", results.save_dir)
    return results, run_name
