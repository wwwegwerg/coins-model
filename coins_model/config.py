DATA_YAML = "dataset_coins/data.yaml"

# Режимы:
# "train"    -> новая тренировка с нуля от pretrained модели
# "resume"   -> продолжить незавершенный run через resume=True
# "finetune" -> дообучение завершенного run как новый запуск от checkpoint
MODE = "train"

# Можно выбрать: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
MODEL_NAME = "yolo26n.pt"

# Путь к checkpoint для resume/finetune. Должен указывать на runs/train/<run_name>/weights/last.pt или best.pt
CKPT_PATH = ""

IMGSZ = 640
BATCH = 12
WORKERS = 2
DEVICE = "mps"

# Для train и finetune это число эпох текущего запуска
EPOCHS = 10

# Для resume это ОБЩЕЕ число эпох после продолжения.
# Например, если было 10 и хочешь еще 10, ставь 20.
RESUME_TOTAL_EPOCHS = 20
