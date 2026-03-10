import joblib

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.MLP.MLP_model import MLPModel
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.base_step import PipelineStep


class TrainingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        self.use_hyperparmeter_tuning = config.pipeline.use_hyperparameter_tuning
        
    def execute(self, step_result: StepResult) -> StepResult:

        model_instance: BaseModel = step_result.model_instance
        
        if self.use_hyperparmeter_tuning:
            model_instance.tune_hyperparameters(step_result)
            
        model_instance.fit(step_result)
        
        # load already fitted model
        # step_result.model_instance = joblib.load("results/LinearRegression/10/092641_09032026_LinearRegression/model.pkl")
        
        step_result.y_test, step_result.y_test_pred, step_result.y_train, step_result.y_train_pred = model_instance.predict(step_result)
            
        print("Training completed.")
        return step_result