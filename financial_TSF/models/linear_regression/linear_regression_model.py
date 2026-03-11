from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel




class LinearRegressionModel(BaseModel):
    def __init__(self, config: AppConfig):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()

    def fit(self, step_result: StepResult):
        
        
        # final refit on combined train+val, including it gives the final model more data before test evaluation.
        X_combined = pd.concat([step_result.X_train, step_result.X_val])
        y_combined = pd.concat([step_result.y_train, step_result.y_val])
        self.model.fit(X_combined, y_combined)
        print("[Ridge] Fitted on combined train+val.")

    def predict(self, step_result: StepResult):
                
        y_test_pred = self.model.predict(step_result.X_test)
        y_train_pred = self.model.predict(step_result.X_train)

        
        return step_result.y_test, y_test_pred, step_result.y_train, y_train_pred

    def tune_hyperparameters(self, step_result: StepResult) -> dict:
        """
        search over different alpha values
        Fits on X_train/y_train, evaluates on X_val/y_val
        sets model to best alpha
        """
        print("[Ridge] Starting hyperparameter tuning...")

        X_train = step_result.X_train
        y_train = step_result.y_train
        X_val = step_result.X_val
        y_val = step_result.y_val

        # alpha search space log-spaced from 0.001 to 10000
        alphas = np.logspace(-3, 4, 50)

        best_alpha = 1.0
        best_mse = float("inf")
        results: list[dict] = []

        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)
            results.append({"alpha": alpha, "val_mse": mse})

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        print(f"[Ridge] Best alpha={best_alpha:.6f}  val_MSE={best_mse:.6f}")
        for result in results:
            print(f"  alpha={result['alpha']:>12.6f}  val_MSE={result['val_mse']:.6f}")

        # update model with best alpha
        self.model = Ridge(alpha=best_alpha)
        print(f"[Ridge] Model updated with alpha={best_alpha:.6f}")

        return {"alpha": best_alpha, "val_mse": best_mse}
