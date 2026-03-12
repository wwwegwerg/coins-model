from ultralytics import YOLO

from ..config import CKPT_PATH, RESUME_TOTAL_EPOCHS
from ..shared import extract_run_name


def run_resume():
    model = YOLO(CKPT_PATH)

    results = model.train(
        resume=True,
        epochs=RESUME_TOTAL_EPOCHS,
    )

    print("Незавершенный run продолжен.")
    print("Результаты сохранены в:", results.save_dir)
    # Для resume имя run уже существует, извлекаем его из save_dir
    run_name = extract_run_name(results.save_dir)
    return results, run_name
