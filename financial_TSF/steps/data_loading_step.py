from pandas import DataFrame
import pandas as pd
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
from pathlib import Path



class DataLoadingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.csv_path = config.data.concatenated_csv
        self.use_all_tickers = config.data.use_all_tickers

    def execute(self, step_result: StepResult) -> StepResult:
        # Implement data loading logic here
        print("Executing Data Loading Step with csv_path:", self.csv_path, "and use_all_tickers:", self.use_all_tickers)
        
        csv_file_paths = Path(self.csv_path).rglob('*.csv')
        
        tickers: list[DataFrame] = []          
                    
        for csv_path in csv_file_paths:
            df = DataFrame()
            try:
                df = pd.read_csv(csv_path)
                # Add ticker from file name AAPL_intraday....csv to df
                file_name = Path(csv_path).stem
                df["ticker"] = file_name.split('_')[0]
                
                tickers.append(df)
                print(f"Loaded data from {csv_path}, shape: {df.shape}")
            except Exception as e:
                print(f"Failed to load data from {csv_path}: {e}")
        step_result.data = tickers
        return step_result