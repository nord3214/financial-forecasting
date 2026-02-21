from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.steps.base_step import PipelineStep


# class ModelChangesStep(PipelineStep):
#     def __init__(self, config: AppConfig) -> None:
#         self.config: AppConfig = config
#         self.model_type: str = config.model.model_type

#     def execute(self, step_result):
            
#             return step_result
            
#         else: 
#             print(f"No seperate model changes step needed for {self.model_type} model, skipping.")
            
#             return step_result
