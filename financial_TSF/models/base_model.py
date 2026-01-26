from abc import ABC, abstractmethod
from typing import Any
from financial_TSF.data_models.step_datamodel import StepResult


class BaseModel(ABC):
    
    @abstractmethod
    def fit(self, step_result: StepResult) -> Any:
        """The actual model should be trained in this method"""

    @abstractmethod
    def predict(self, step_result: StepResult) -> tuple:
        """Generate predictions for the provided dataset.
        
        Returns:
            Tuple containing:
                y_test: Ground-truth values for the test split.
                y_test_pred: Model predictions for the test split.
                y_train: Ground-truth values for the training split.
                y_train_pred: Model predictions for the training split.
        """
        
    def tune_hyperparameters(self, step_result: StepResult) -> dict | None:
        """Optional method to tune hyperparameters of the model.
        Returns:
            A dictionary containing the best hyperparameters found.
        """
        pass