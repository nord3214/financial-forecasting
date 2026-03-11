import json
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel
from scipy.stats import randint

from financial_TSF.utils.utils import get_result_path, save_param_dict, print_target_diagnostics, plot_learning_curve



HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "random_state": 17,
    "n_jobs": -1,          # n_jobs=-1 copies X_train per worker on Windows (spawn) → OOM; keep 1
    "max_features": "sqrt",
    "min_samples_leaf": 40,
    "max_depth": 15,
}

class RandomForestModel(BaseModel):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.estimator: Optional[Any] = None  # placeholder fitted estimator model
        
        self.X_train: Optional[pd.DataFrame]
        self.X_test: Optional[pd.DataFrame] 
        self.y_train: Optional[pd.Series] 
        self.y_test: Optional[pd.Series] 
        
    def fit(self, step_result: StepResult):
        
        # self.X_train = step_result.X_train
        # self.X_test = step_result.X_test
        # self.y_train = step_result.y_train
        # self.y_test = step_result.y_test
        
        if step_result.X_train is None or step_result.y_train is None:
            raise RuntimeError("Data splits are not prepared. Call prepare_data() or Datasplit Pipeline step first.")

        print_target_diagnostics(step_result.y_train)

        from sklearn.ensemble import RandomForestRegressor

        # fit on train only with oob_score — oob_prediction_ is a free generalisation estimate
        rfr = RandomForestRegressor(criterion="squared_error", oob_score=True)
        rfr.set_params(**HYPERPARAMS)

        # --- n_estimators learning curve (warm_start = one growing forest, no extra RAM) ---
        n_total = HYPERPARAMS.get("n_estimators", 300)
        n_steps = 10
        step_size = max(1, n_total // n_steps)
        checkpoints = list(range(step_size, n_total + 1, step_size))
        if checkpoints[-1] != n_total:
            checkpoints.append(n_total)

        rfr.set_params(warm_start=True)
        train_maes, val_maes = [], []
        for n in checkpoints:
            rfr.set_params(n_estimators=n)
            rfr.fit(step_result.X_train, step_result.y_train)
            train_maes.append(mean_absolute_error(step_result.y_train, rfr.predict(step_result.X_train)))
            val_maes.append(mean_absolute_error(step_result.y_val,     rfr.predict(step_result.X_val)))
            print(f"[RF] n_trees={n:>4}  train MAE={train_maes[-1]:.6f}  val MAE={val_maes[-1]:.6f}")
        rfr.set_params(warm_start=False)

        save_path = get_result_path(self.config, step_result)
        plot_learning_curve(
            train_maes, val_maes,
            model_name="RF",
            save_path=save_path,
            xlabel="Number of Trees",
            x_values=checkpoints,
        )
        # --- final diagnostics at full n_estimators ---
        self.estimator = rfr
        print("[RF] Fitted on X_train.")
        oob_mae = mean_absolute_error(step_result.y_train, rfr.oob_prediction_)
        print(f"[RF] train MAE: {train_maes[-1]:.6f} | val MAE: {val_maes[-1]:.6f} | OOB MAE: {oob_mae:.6f}")

        save_param_dict(
            {**HYPERPARAMS, "train_mae": train_maes[-1], "val_mae": val_maes[-1], "oob_mae": oob_mae},
            self.config, step_result,
        )

        importances = pd.Series(rfr.feature_importances_, index=step_result.X_train.columns)
        importances = importances.sort_values(ascending=False)
        print("Feature importances:\n", importances)

        self.save_model(step_result)
            
            
        
    def predict(self, step_result: StepResult) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        # load already fitted model
        # if self.estimator is None:
        #     self.load_model(Path("resultsTraining/RF/10_20_30/145324_10032026_RF/random_forest_model.joblib"))

        if self.estimator is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        if step_result.X_test is None or step_result.X_train is None or step_result.y_test is None or step_result.y_train is None:
            raise RuntimeError("Data splits are not prepared. Call prepare_data()/fit() first.")

        y_test_pred = self.estimator.predict(step_result.X_test)
        y_train_pred = self.estimator.predict(step_result.X_train)
        
        # calculate mean and variance only distribution of epistemic uncertainty
        
        # tree_preds = np.array([tree.predict(self.X_test) for tree in self.estimator.estimators_])
        # mean_pred = np.mean(tree_preds, axis=0)
        # variance_pred = np.var(tree_preds, axis=0)
        
        # feature importance 
        



        return step_result.y_test, y_test_pred, step_result.y_train, y_train_pred

    def tune_hyperparameters(self, step_result: StepResult) -> dict:
        """
        random search over hyperparameters using (X_val / y_val) and 
        update HYPERPARAMS global dict with best config for subsequent fit() call
        """
        from sklearn.ensemble import RandomForestRegressor as _RFR

        print("[RF] Starting hyperparameter tuning with validation set...")

        X_train = step_result.X_train
        y_train = step_result.y_train
        X_val   = step_result.X_val
        y_val   = step_result.y_val


        # search space
        param_grid = {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [10, 15, 20, 30, None],
            "min_samples_leaf": [10, 20, 40, 60, 100],
            "min_samples_split":[2, 5, 10, 20],
            "max_features":     ["sqrt", "log2", 0.3, 0.5, None],
        }

        n_iter = 10  # number of combinations 
        rng = np.random.RandomState(1337)

        best_mse = float("inf")
        best_params: dict = {}
        results: list[dict] = []

        for i in range(n_iter):
            # sample one random config dict
            params = {k: rng.choice(v) for k, v in param_grid.items()}
            print("[RF] current training with params: ", params)

            rfr = _RFR(
                **params,
                n_jobs=-2,
                random_state=1337,
                criterion="squared_error",
            )
            rfr.fit(X_train, y_train)
            val_pred = rfr.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)

            results.append({**params, "val_mse": mse})
            print(f"  [{i+1:>2}/{n_iter}] val_MSE={mse:.6f}  {params}")

            if mse < best_mse:
                best_mse = mse
                best_params = params

        print(f"\n[RF] Best val_MSE={best_mse:.6f}")
        print(f"[RF] Best params: {best_params}")
        
        result_dir: Path = get_result_path(self.config, step_result) / "hyperparameter_tuning"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame(results).to_csv(result_dir / f"hyperparameter_tuning_results_for{self.config.features.complexity_levels}.csv", index=False)
        
        for param in best_params:
            val = best_params[param]
            if isinstance(val, np.generic):
                best_params[param] = val.item()
            
        with open(result_dir / f"best_params_for{self.config.features.complexity_levels}.json", "w") as f:
            json.dump(best_params, f, indent=4)
            print(f"[RF] Hyperparameter tuning results saved to {result_dir / f'best_params_for{self.config.features.complexity_levels}.json'}")

        #update global HYPERPARAMS with best params
        HYPERPARAMS.update(best_params)

        return {"best_params": best_params, "val_mse": best_mse}
    
    def save_model(self, step_result: StepResult) -> Path:
        """Persist the fitted estimator to disk."""
        if self.estimator is None:
            raise RuntimeError("No fitted estimator to save. Call fit() first.")

        result_dir = get_result_path(self.config, step_result) / "saved_model"
        result_dir.mkdir(parents=True, exist_ok=True)
        model_path = result_dir / "random_forest_model.joblib"
        joblib.dump(self.estimator, model_path, compress=3)
        print(f"[RF] Model saved to {model_path}")
        return model_path

    def load_model(self, model_path: Path) -> None:
        """Load a previously saved estimator."""
        self.estimator = joblib.load(model_path)
        print(f"[RF] Model loaded from {model_path}")