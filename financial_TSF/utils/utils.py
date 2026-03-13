from pathlib import Path
import json
import torch
import pandas as pd
from matplotlib import pyplot as plt

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult




def plot_learning_curve(train_losses, val_losses, model_name="Modell", save_path: Path | None = None, xlabel: str = "Epochs", x_values=None) -> None:
    x = x_values if x_values is not None else range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(x, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(x, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'Learning Curves: {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Loss MAE', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        save_path = save_path / "plots"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'{model_name}_learning_curve.png', dpi=100)
        plt.close()
    else:
        plt.show()


def print_target_diagnostics(y_train: pd.DataFrame) -> None:
    y_np = y_train.to_numpy()
    print(f"[Target Diagnostics] y_train mean:  {y_np.mean():.6f}")
    print(f"[Target Diagnostics] y_train std:   {y_np.std():.6f}")
    print(f"[Target Diagnostics] y_train min:   {y_np.min():.6f}")
    print(f"[Target Diagnostics] y_train max:   {y_np.max():.6f}")
    print(f"[Target Diagnostics] % near zero (|y|<0.01): {(abs(y_np) < 0.01).mean()*100:.1f}%")


def print_prediction_diagnostics(model: torch.nn.Module, val_loader, train_loader, device: torch.device, epoch: int, forward_fn=None) -> None:
    """
    forward_fn: optional callable(model, inputs) -> preds tensor.
    If None, uses model(inputs) directly (MLP-style).
    For PatchTST pass: forward_fn=lambda m, x: m(past_values=x).regression_outputs
    """
    if forward_fn is None:
        forward_fn = lambda m, x: m(x)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device).float()
            all_preds.append(forward_fn(model, inputs).cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    print(f"--- diagnosis for epoch {epoch} ---")
    print(f"Target Std:     {all_targets.std().item():.4f}")
    print(f"Prediction Std: {all_preds.std().item():.4f}")
    print(f"Prediction Mean:{all_preds.mean().item():.4f}")

    with torch.no_grad():
        sample_inputs, _ = next(iter(train_loader))
        sample_inputs = sample_inputs.to(device).float()
        sample_preds = forward_fn(model, sample_inputs)
        print(f"Train-sample Pred Std: {sample_preds.std().item():.4f}, Mean: {sample_preds.mean().item():.4f}")


def get_result_path(config: AppConfig, step_result: StepResult)-> Path:
    
    save_path = Path(config.results.results_path)
    
    timestamp = step_result.start_time.strftime("%H%M%S_%d%m%Y") 
    feature_complexity_str = "_".join(map(str, config.features.complexity_levels))
    
    run_dir: Path = save_path / config.model.model_type / feature_complexity_str / f"{timestamp}_{config.model.model_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir

def save_param_dict(params: dict, config: AppConfig, step_result: StepResult) -> None:
    save_path = get_result_path(config, step_result)
    param_file = save_path / "params.txt"
    with open(param_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def save_tuning_results(
    all_trials: list[dict],
    final_params: dict,
    model_name: str,
    config: AppConfig,
    step_result: StepResult,
) -> None:
    """Save hyperparameter tuning results to JSON

    Parameters
    -
    all_trials : list[dict]
        "params" (dict) and "val_loss" (float)
    final_params : dict
       the best params.
    model_name : str
        e.g. "MLP", "RF", "PatchTST"
    config, step_result : used to resolve the output directory.
    """

    sorted_trials = sorted(all_trials, key=lambda x: x["val_loss"])
    best_val_loss = sorted_trials[0]["val_loss"] if sorted_trials else None

    output = {
        "model": model_name,
        "n_trials": len(all_trials),
        "best_val_loss": best_val_loss,
        "final_params": final_params,
        "all_trials_sorted_by_val_loss": [
            {"rank": rank + 1, "val_loss": t["val_loss"], "params": t["params"]}
            for rank, t in enumerate(sorted_trials)
        ],
    }

    save_path = get_result_path(config, step_result) / "hyperparameter_tuning"
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / "tuning_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=4, default=str)
    print(f"[{model_name}] Tuning results saved to {out_file}")
    