from dataclasses import dataclass, field
from typing import Any
from pandas import DataFrame

from financial_TSF.models.base_model import BaseModel

@dataclass
class StepResult:
    data: list[DataFrame] = field(default_factory=list)
    model_instance: Any = None
    trained_model: Any = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    y_test: Any = None
    y_test_pred: Any = None
    y_train: Any = None
    y_train_pred: Any = None