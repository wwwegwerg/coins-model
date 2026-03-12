from pathlib import Path
from coins_model.app import export_model, extract_run_name, validate_best_model

results = "/Users/bdt/code/coins-model/runs/train/train_20260312_182217"
run_name = extract_run_name(results)
best_path = Path(results) / "weights" / "best.pt"
print(results, run_name, best_path)
best_model = validate_best_model(best_path, run_name)
export_model(best_model, run_name)