import json
from typing import Any, Dict
from financial_TSF.data_models.config_data_model import AppConfig, FeaturesConfig
from financial_TSF.features.technical_indicator import TechnicalIndicators
from financial_TSF.steps.base_step import PipelineStep


class FeatureStep(PipelineStep):
    
    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        with open(self.config.features.feature_config_path, "r") as f:
            self.feature_config: Dict[str, Any] = json.load(f)
            # print("FeatureStep configuration loaded:", self.feature_config)
    
    
    def execute(self, data, model):
        TechnicalIndicators(self.config.features.feature_config_path, data)
        # return super().execute(data, model)
    
    
