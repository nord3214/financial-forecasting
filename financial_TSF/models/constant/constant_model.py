
from pathlib import Path
from typing import Any

from sklearn.dummy import DummyRegressor
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel


class ConstantModel(BaseModel):
    def __init__(self, config: AppConfig):
        # self.model = Ridge(alpha=1.0)
        self.model = DummyRegressor(strategy="median")

    def fit(self, step_result: StepResult):
        # fit on X_train only — avoids duplicating data in RAM
        self.model.fit(step_result.X_train, step_result.y_train)
        print("[DummyRegressor] Fitted on X_train.")


    def predict(self, step_result: StepResult):
                
        y_test_pred = self.model.predict(step_result.X_test)
        y_train_pred = self.model.predict(step_result.X_train)

        
        return step_result.y_test, y_test_pred, step_result.y_train, y_train_pred
    
    def save_model(self, step_result: StepResult) -> Any:
        pass
    
    def load_model(self, model_path: Path) -> Any: 
        pass