from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel


class NaiveModel(BaseModel):
    """naive baseline.

    predict()  each prediction is the previous actual value.
                For the first test sample the last training value is used.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def fit(self, step_result: StepResult):
        pass

    def predict(self, step_result: StepResult):
        y_train_np = step_result.y_train.to_numpy()  # (N_train, H)
        y_test_np  = step_result.y_test.to_numpy()   # (N_test,  H)

        #test predictions: y(t) = y(t-1) 
        # prepend the last training row so the shift covers the first test step
        test_prev = np.concatenate([y_train_np[-1:], y_test_np[:-1]], axis=0)
        y_test_pred = test_prev  # (N_test, H)

        #train predictions: y(t) = y(t-1) 
        # first train sample has no predecessor - use itself
        train_prev = np.concatenate([y_train_np[:1], y_train_np[:-1]], axis=0)
        y_train_pred = train_prev  # (N_train, H)

        test_mae  = mean_absolute_error(y_test_np,  y_test_pred)
        train_mae = mean_absolute_error(y_train_np, y_train_pred)
        print(f"[Naive] train MAE: {train_mae:.6f} | test MAE: {test_mae:.6f}")

        return step_result.y_test, y_test_pred, step_result.y_train, y_train_pred

    def tune_hyperparameters(self, step_result: StepResult):
        pass
    
    def load_model(self, model_path: Path):
        pass
    
    def save_model(self, step_result: StepResult):
        pass