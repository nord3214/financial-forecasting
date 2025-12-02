from abc import ABC, abstractmethod


class BaseModel(ABC):
    
    @abstractmethod
    def fit(self, data):
        """The actual model should be trained in this method"""

    @abstractmethod
    def predict(self, data) -> tuple:
        """Generate predictions for the provided dataset.
        
        Returns:
            Tuple containing:
                y_test: Ground-truth values for the test split.
                y_test_pred: Model predictions for the test split.
                y_train: Ground-truth values for the training split.
                y_train_pred: Model predictions for the training split.

        """