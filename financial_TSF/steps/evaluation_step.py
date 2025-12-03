from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.base_step import PipelineStep


class EvaluationStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:
        
        # machine learning model evaluation

        print("Metrics on Test Set:")
        print("Mean Absolute Error (MAE): ", mean_absolute_error(step_result.y_test, step_result.y_test_pred))
        print("Mean Sqaured Error (MSE): ", mean_squared_error(step_result.y_test, step_result.y_test_pred))
        print("R2 Score: ", r2_score(step_result.y_test, step_result.y_test_pred))  
        
        print("Metrics on Train Set:")
        print("Mean Absolute Error (MAE): ", mean_absolute_error(step_result.y_train, step_result.y_train_pred))
        print("Mean Sqaured Error (MSE): ", mean_squared_error(step_result.y_train, step_result.y_train_pred))
        print("R2 Score: ", r2_score(step_result.y_train, step_result.y_train_pred))  
        
        # economic model evaluation
        
        
        return step_result