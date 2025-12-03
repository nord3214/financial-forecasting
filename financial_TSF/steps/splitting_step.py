from sklearn.model_selection import train_test_split
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.RF.random_forest_regressor import SETTINGS
from financial_TSF.steps.base_step import PipelineStep


class SplittingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:

        # :TODO change to adjust for multiple dataframes

        df = step_result.data[0].copy()
        
        # :TODO check for memory if large datasets clear data
        # step_result.data = list()
        
        
        df["target"] = df["close"].shift(-SETTINGS["forecast_horizon"])
        df = df.dropna()
        df = df.drop("ticker", axis=1)

        X = df.drop("target", axis=1)    
        y = df["target"]
        
        step_result.X_train, step_result.X_test, step_result.y_train, step_result.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        
        return step_result
        
