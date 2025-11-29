

import json

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.pipeline.registry import STEP_REGISTRY
from financial_TSF.steps.base_step import PipelineStep


class Pipeline:
    def __init__(self, config_path: str = "financial_TSF/config.json"):
        with open(config_path, "r") as f:
            self.config: AppConfig = AppConfig.from_dict(json.load(f))
            
        self.steps: list[PipelineStep] = [
            # STEP_REGISTRY[step_name](self.config) for step_name in self.config["pipeline"]["steps"]]
            STEP_REGISTRY[step_name](self.config) for step_name in self.config.pipeline.steps]
        
        
    def run(self, data):
        for step in self.steps:
            print(f"Executing step: {step.__class__.__name__}")
            data = step.execute(data, model=None)
        return data
        
        
    