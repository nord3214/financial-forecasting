from abc import ABC, abstractmethod
from typing import Any

class PipelineStep(ABC):
    @abstractmethod
    def execute(self, data: Any, model: Any) -> Any:
        """Execute the step with the given data and model if possible return the data and model for next step"""