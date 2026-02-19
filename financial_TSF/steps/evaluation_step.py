from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd

from financial_TSF.utils.economic_evaluation.top_k_simulator import TopKSimulator



class EvaluationStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:
        
        step_result = self.map_predictions_to_tickers(step_result)
                
        # do whole evaluation here and save metrics in step_result.metrics
        step_result.metrics = self.compute_multistep_metrics(step_result.y_test, step_result.y_test_pred, step_result)
        
        
        
        return step_result
    
    def map_predictions_to_tickers(self, step_result: StepResult) -> StepResult:
        """
            Map global y_test_pred to each ticker's in split_data y_test_pred
        """
    
        start_idx = 0
        for split in step_result.split_data:
            n_rows = split['y_test'].shape[0]
            current_slice = step_result.y_test_pred[start_idx:start_idx + n_rows]
            
            # assign the y_pred slice to the corresponding ticker in split_data
            split['y_test_pred'] = pd.DataFrame(
                current_slice,
                columns=split['y_test'].columns,
                index=split['y_test'].index
            )
            
            # validate for using correct prediction
            if not split['y_test_pred'].shape == split['y_test'].shape:
                raise ValueError("Shape mismatch between y_test and y_test_pred in split_data mapping")
            if not (split['y_test'].index == split['y_test_pred'].index).all():
                raise ValueError("Index mismatch between y_test and y_test_pred in split_data mapping")
            # compare if y_test values are the same - to induce operating on the correct ticker slices
            if not (split['y_test'].values == step_result.y_test[start_idx:start_idx + n_rows].values).all():
                raise ValueError("Probably ticker mismatch between y_test in split_data and global y_test in StepResult")

            start_idx += n_rows
            
        return step_result
   
    def compute_multistep_metrics(self, y_true, y_pred, step_result: StepResult = StepResult()) -> dict[str, float]:
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
                     
            
            # ------------------------- economic evaluation -------------------------
        
            sim = TopKSimulator(step_result.split_data, transaction_cost_bps=
                                
                                
                                1.0)
            sim.prepare_data(horizon_col=f"target_{self.config.settings.forecast_horizon[i]}")
            
            sim.run_simulation(k=5)

            # sim.plot_equity_curve()
            
        
            metrics[f"Horizon_{horizon}"] = {
                "MAE": mean_absolute_error(y_true_horizon, y_pred_horizon),
                "MSE": mean_squared_error(y_true_horizon, y_pred_horizon),
                "RMSE": root_mean_squared_error(y_true_horizon, y_pred_horizon),
                "R2": r2_score(y_true_horizon, y_pred_horizon),
                "Hit_Rate": float(sim.hit_rate),
                "Sharpe_Ratio": float(sim.annualized_sharpe),
                "Percentage_Return": float(sim.percentage_return),
                "Number_of_Trades": int(sim.no_of_entries)
            }
            
        metrics["Global"] = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }
        
        print("Multi-step Forecasting Metrics: ", metrics)    
        return metrics
    

