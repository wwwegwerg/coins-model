# Coins YOLO

Проект запускается через `main.py`, но внутренняя логика теперь разделена по Python-модулям.

## Режимы

В `coins_model/config.py` есть параметр `MODE` с тремя режимами:

- `"train"` - тренировка новой модели от pretrained-весов
- `"resume"` - продолжение незавершенного run через `resume=True`
- `"finetune"` - дообучение от указанного checkpoint как новый запуск

Там же задаются основные параметры:

- `MODEL_NAME`
- `CKPT_PATH`
- `EPOCHS`
- `RESUME_TOTAL_EPOCHS`
- `IMGSZ`
- `BATCH`
- `WORKERS`
- `DEVICE`

## Структура

```text
coins-model/
  main.py
  coins_model/
    app.py
    config.py
    shared.py
    evaluate.py
    export.py
    modes/
      train.py
      resume.py
      finetune.py
```

## Структура артефактов (runs/)

Все результаты обучения, валидации и экспорта сохраняются в папке `runs/` с единой структурой:

```text
runs/
  train/
    train_YYYYMMDD_HHMMSS/     # Артефакты тренировки
    finetune_YYYYMMDD_HHMMSS/  # Артефакты дообучения
  val/
    train_YYYYMMDD_HHMMSS/   # Результаты валидации для train
    finetune_YYYYMMDD_HHMMSS/# Результаты валидации для finetune
  export/
    train_YYYYMMDD_HHMMSS/   # ONNX-модели
    finetune_YYYYMMDD_HHMMSS/# ONNX-модели после finetune
```

Каждый запуск получает уникальное имя с префиксом режима и timestamp. Это имя используется для связывания артефактов тренировки, валидации и экспорта.

**Важно:** Не переименовывайте папки в `runs/` вручную — это нарушит связь между чекпоинтами и их артефактами.

## Запуск

1. Установить `uv`
2. Проверить параметры в `coins_model/config.py`:
   - `MODE` — режим работы (`train`, `resume`, `finetune`)
   - `CKPT_PATH` — путь к checkpoint (для `resume`/`finetune`), должен указывать на `runs/train/<run_name>/weights/last.pt` или `best.pt`
   - Остальные параметры обучения
3. В директории проекта выполнить:

```bash
uv sync
uv run main.py
```
