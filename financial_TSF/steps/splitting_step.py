import pandas as pd
from sklearn.model_selection import train_test_split
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.rf.random_forest_regressor import SETTINGS
from financial_TSF.steps.base_step import PipelineStep


class SplittingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        self.use_log_returns = True
        
    def execute(self, step_result: StepResult) -> StepResult:

        # :TODO maybe add index again for time series split in batches (lstm, transformer...)

        df = pd.concat(step_result.data, axis = 0, ignore_index=True)
        # df = step_result.data[0].copy()
        
        # :TODO check for memory if large datasets clear data
        # step_result.data = list()
        
        
        df = df.drop("ticker", axis=1)

        if self.use_log_returns:

            for i, horizon in enumerate(self.config.settings.forecast_horizon):
                df[f"target{i+1}"] = df['Log_Return_1'].rolling(window=horizon).sum().shift(-horizon)
            
        else:
            # create target-columns from horizons dynamically
            for i, horizon in enumerate(self.config.settings.forecast_horizon):
                df[f"target{i+1}"] = df["close"].shift(-horizon)
            
        df = df.dropna()
        
        y = df.iloc[:, -len(self.config.settings.forecast_horizon):]
        X = df.iloc[:, :-len(self.config.settings.forecast_horizon)]
        
        
        step_result.X_train, step_result.X_test, step_result.y_train, step_result.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        
        return step_result
        
