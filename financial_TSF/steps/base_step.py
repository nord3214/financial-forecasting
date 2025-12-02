from abc import ABC, abstractmethod
from typing import Any

from financial_TSF.data_models.step_datamodel import StepResult

class PipelineStep(ABC):
    @abstractmethod
    def execute(self, step_result: StepResult) -> Any:
        """Execute the step with the given data and model if possible return the data and model for next step"""