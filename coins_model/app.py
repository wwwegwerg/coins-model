from .config import MODE
from .evaluate import validate_best_model
from .export import export_onnx
from .modes import run_finetune, run_resume, run_train
from .shared import validate_mode


def main():
    validate_mode(MODE)

    if MODE == "train":
        results, run_name = run_train()
    elif MODE == "resume":
        results, run_name = run_resume()
    else:
        results, run_name = run_finetune()

    best_path = f"{results.save_dir}/weights/best.pt"
    best_model = validate_best_model(best_path, run_name)
    export_onnx(best_model, run_name)
