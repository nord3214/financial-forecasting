from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.MLP.MLP_model import MLP, MLPModel
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.base_step import PipelineStep


class TrainingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:

        model_is_deep = False

        if model_is_deep:
            mlp = MLP()
            model_instance = MLPModel(mlp)

            model_instance.fit(step_result)

        else:
            model_instance: BaseModel = step_result.model_instance
            model_instance.fit(step_result)
            step_result.y_test, step_result.y_test_pred, step_result.y_train, step_result.y_train_pred = model_instance.predict(step_result)
            
        print("Training completed.")
        return step_result