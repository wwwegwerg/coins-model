# Coins YOLO

## Структура

```text
coins-model/
  main.py
  coins_model/
    app.py
    config.py
```

Вся runtime-логика находится в `coins_model/app.py`, а все настраиваемые параметры вынесены в `coins_model/config.py`.

## Конфиг

В `coins_model/config.py` используются 5 блоков:

- `PATHS` - куда сохраняются артефакты
- `RUN` - режим, модель, checkpoint и количество эпох
- `RUN_PREFIXES` - префиксы имён запусков
- `TRAIN` - параметры `model.train(...)`
- `VALIDATE` - параметры `model.val(...)`
- `EXPORT` - параметры `model.export(...)`

### Режимы

`RUN["mode"]` может быть одним из значений:

- `"train"` - новая тренировка от pretrained-весов
- `"resume"` - продолжение незавершённого run через `resume=True`
- `"finetune"` - дообучение от checkpoint как новый запуск

### Что обычно редактировать

- `RUN["mode"]`
- `RUN["model_name"]`
- `RUN["checkpoint_path"]`
- `RUN["epochs"]`
- `RUN["resume_total_epochs"]`
- `TRAIN["imgsz"]`
- `TRAIN["batch"]`
- `TRAIN["workers"]`
- `TRAIN["device"]`
- `TRAIN["amp"]`
- `TRAIN["optimizer"]`
- `TRAIN["patience"]`
- `TRAIN["save_period"]`
- `TRAIN["cache"]`
- `EXPORT["format"]`

## Структура `runs/`

Теперь артефакты должны складываться в предсказуемые директории без вложенного `runs/detect/...`:

```text
runs/
  train/
    train_YYYYMMDD_HHMMSS/
    finetune_YYYYMMDD_HHMMSS/
  val/
    train_YYYYMMDD_HHMMSS/
    finetune_YYYYMMDD_HHMMSS/
  export/
    train_YYYYMMDD_HHMMSS/
    finetune_YYYYMMDD_HHMMSS/
```

Если использовать старые checkpoint из уже созданных каталогов с `runs/detect/...`, режим `resume` продолжит работать с ними как есть, но новые `train` и `finetune` должны сохраняться уже в нормальную структуру.

## Запуск

1. Установить `uv`.
2. Положить `dataset_coins` в корень проекта.
3. Настроить `coins_model/config.py`.
4. Запустить:

```bash
uv sync
uv run main.py
```
