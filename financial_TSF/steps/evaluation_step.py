from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd
import numpy as np


class EvaluationStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:
        
        # machine learning model evaluation

        print("Metrics on Test Set:")
        print("Mean Absolute Error (MAE): ", mean_absolute_error(step_result.y_test, step_result.y_test_pred))
        print("Mean Sqaured Error (MSE): ", mean_squared_error(step_result.y_test, step_result.y_test_pred))
        print("R2 Score: ", r2_score(step_result.y_test, step_result.y_test_pred))  
        
        step_result.metrics = self.compute_multistep_metrics(step_result.y_test, step_result.y_test_pred)
        
        
        # print("Metrics on Train Set:")
        # print("Mean Absolute Error (MAE): ", mean_absolute_error(step_result.y_train, step_result.y_train_pred))
        # print("Mean Sqaured Error (MSE): ", mean_squared_error(step_result.y_train, step_result.y_train_pred))
        # print("R2 Score: ", r2_score(step_result.y_train, step_result.y_train_pred))  
        
        # # economic model evaluation
        # # :TODO needs full rework of economic evaluation maybe with backtrader/ PyPortfolioOpt library
        # df = pd.DataFrame()
        # #current price
        # df["price"] = step_result.X_test["close"]
        # # future price +12 steps
        # df["future_price"] = step_result.y_test
        
        # df["real_return"] = df["future_price"] - df["price"]
        # df["prediction"] = np.sign(step_result.y_test_pred - df["future_price"])
        # df["strategy_return"] = df["real_return"] * df["prediction"]
        
        # total_return = df["strategy_return"].sum()
        # print("Total Strategy Return over Test Period: ", total_return)
        
        # # 3. Compute Sharpe (annualized)
        # mean_ret = df['strategy_return'].mean()
        # std_ret = df['strategy_return'].std()

        # # number of trades per day
        # trades_per_day = 13  
        # annualization = np.sqrt(252 * trades_per_day)

        # sharpe = (mean_ret / std_ret) * annualization
        # print("Sharpe Ratio:", sharpe)
        
        
        return step_result
    
    def compute_multistep_metrics(self, y_true, y_pred) -> dict[str, float]:
        """
            H = number of forecast horizons
            y_true: shape (N, H)
            y_pred: shape (N, H)
        """
        metrics = {}
        
        for i, horizon in enumerate(self.config.settings.forecast_horizon):
            # :TODO convert to dataframe or different indexing!
            y_true_horizon = pd.DataFrame(y_true).iloc[:, i]
            y_pred_horizon = pd.DataFrame(y_pred).iloc[:, i]
            
            mae = mean_absolute_error(y_true_horizon, y_pred_horizon)
            mse = mean_squared_error(y_true_horizon, y_pred_horizon)
            r2 = r2_score(y_true_horizon, y_pred_horizon)
            
            metrics[f"Horizon_{horizon}"] = {
                "MAE": mae,
                "MSE": mse,
                "R2": r2
            }
            
        metrics["Global"] = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }
        
        print("Multi-step Forecasting Metrics: ", metrics)    
        return metrics