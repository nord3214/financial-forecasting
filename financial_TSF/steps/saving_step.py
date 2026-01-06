from datetime import datetime
import json
from pathlib import Path
import shutil

import joblib
import pandas as pd
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep


class SavingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:
        
        save_path = Path(self.config.results.results_path)
        
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        run_dir = save_path / f"{timestamp}_{self.config.model.model_type}"
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # save model 
        joblib.dump(step_result.model_instance, run_dir / "model.pkl")
        
        # save pipeline config 
        shutil.copy("financial_TSF/config.json", run_dir / "pipeline_config.json")
        
        # save feature config
        shutil.copy("financial_TSF/features/feature_config.json", run_dir / "feature_config.json")
        
        # save model config
        # :TODO create config file for model hyperparameters
            
        # save metrics :TODO save metrics in step_result
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(step_result.metrics, f, indent=4)
        
            
        # save prediction data
        
        # :TODO handle multi horizon
        pd.DataFrame(step_result.y_test).to_csv(run_dir / "y_test.csv", index=False)
        pd.DataFrame(step_result.y_train).to_csv(run_dir / "y_train.csv", index=False)
        pd.DataFrame(step_result.y_test_pred).to_csv(run_dir / "y_test_pred.csv", index=False)
        pd.DataFrame(step_result.y_train_pred).to_csv(run_dir / "y_train_pred.csv", index=False)
        
        # pd.DataFrame({
        #     "y_true": step_result.y_train,
        #     "y_pred": step_result.y_train_pred
        # }).to_csv(run_dir / "preds_train.csv", index=False)
        
        # pd.DataFrame({
        #     "y_true": step_result.y_test,
        #     "y_pred": step_result.y_test_pred
        # }).to_csv(run_dir / "preds_test.csv", index=False)

        with open(run_dir / "summary.txt", "w") as f:
            f.write(f"Model Type: {self.config.model.model_type}\n")
            f.write(f"Run with {step_result.X_train.shape[1]} features.\n")
            f.write(f"Features complexity level: {self.config.features.complexity_levels}\n")
            f.write(f"Features: {', '.join(step_result.X_train.columns)}\n")
            f.write(f"Run with {step_result.X_train.shape[0]} training samples and {step_result.X_test.shape[0]} test samples.\n")
            f.write(f"Run duration was: {datetime.now() - step_result.start_time} \n")
            f.write(f"Results saved in: {run_dir}\n")
            f.write(f"Trained on {self.config.data.number_of_tickers} tickers. [{', '.join(step_result.ticker_names)}] \nfrom {self.config.data.start_date} to {self.config.data.end_date}\n")
            json.dump(step_result.metrics, f, indent=4)
            
            print("Pipeline finished, results saved to:", run_dir/"summary.txt")
        
        return step_result
        