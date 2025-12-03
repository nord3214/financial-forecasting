from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.base_step import PipelineStep


class TrainingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:
        # :TODO eventually implement splitting logic earlier previous step in pipeline
        data = step_result.data
        print("Received data for Training Step with", len(data), "dataframes.")
        
        model_instance: BaseModel = step_result.model_instance
        model_instance.fit(step_result)
        step_result.y_test, step_result.y_test_pred, step_result.y_train, step_result.y_train_pred = model_instance.predict(step_result)
        
        print("Training completed.")
        step_result.data = data
        return step_result