

from dataclasses import dataclass
import json
from typing import Any

from pandas import DataFrame

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.model_registry import MODEL_REGISTRY
from financial_TSF.pipeline.registry import STEP_REGISTRY
from financial_TSF.steps.base_step import PipelineStep


    
    

class Pipeline:
    def __init__(self, config_path: str = "financial_TSF/config.json"):
        with open(config_path, "r") as f:
            self.config: AppConfig = AppConfig.from_dict(json.load(f))
        self.model = MODEL_REGISTRY[self.config.model.model_type](self.config)
        self.steps: list[PipelineStep] = [
            STEP_REGISTRY[step_name](self.config) for step_name in self.config.pipeline.steps]
        
        
    def run(self, step_result: StepResult):
        step_result.model_instance = self.model
        for step in self.steps:
            print("--------------------------------------------------")
            print(f"Executing step: {step.__class__.__name__}")
            step_result = step.execute(step_result)
        return step_result
        
        
    
    

    