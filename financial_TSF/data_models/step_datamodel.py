from dataclasses import dataclass, field
from typing import Any, TypedDict
from pandas import DataFrame
import numpy.typing as npt

class SplitData(TypedDict):
    # :TODO think about better typing here
    ticker_name: str 
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    x_scaler: Any
    y_scaler: Any

@dataclass
class StepResult:
    data: list[DataFrame] = field(default_factory=list)
    model_instance: Any = None
    data_scalers: list[dict[str, Any]] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    split_data: list[SplitData] = field(default_factory=list)
    X_test: Any = None
    y_test: Any = None
    y_test_pred: Any = None
    X_train: Any = None
    y_train: Any = None
    y_train_pred: Any = None
    metrics: dict[str, Any] = field(default_factory=dict),
    ticker_names: list[str] = field(default_factory=list)
    start_time: Any = None