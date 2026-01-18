from pandas import DataFrame
import pandas as pd
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
from pathlib import Path



class DataLoadingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.csv_path = config.data.concatenated_csv
        self.number_of_tickers = config.data.number_of_tickers
        self.start_date = config.data.start_date
        self.end_date = config.data.end_date
        self.use_cached_processed_data = config.data.use_cached_processed_data

    def execute(self, step_result: StepResult) -> StepResult:
        # Implement data loading logic here
        
        print("Executing Data Loading Step with csv_path:", self.csv_path, "and number_of_tickers:", self.number_of_tickers)
            
        csv_file_paths: list[Path] = list(Path(self.csv_path).rglob('*.csv'))
        
        if self.number_of_tickers:
            csv_file_paths = list(csv_file_paths)[:self.number_of_tickers]     
            
        # extract the ticker names from the file names
        ticker_names = [csv_path.name.split('_')[0] for csv_path in csv_file_paths]
        step_result.ticker_names = ticker_names      
        
        if self.use_cached_processed_data:
            print("Using cached processed data, skipping data loading step.")
            return step_result
        
        else: 
            
            df_tickers: list[DataFrame] = []     
            for csv_path in csv_file_paths:
                df = DataFrame()
                try:
                    df = pd.read_csv(csv_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')

                    df = df.loc[self.start_date:self.end_date]
                    
                    # Add tickername from file name AAPL_intraday....csv to df
                    file_name = Path(csv_path).stem
                    df["ticker"] = file_name.split('_')[0]
                    
                    if not self.__validate_data(df):
                        raise Exception("Data validation failed for file: " + str(csv_path))
                    
                    df_tickers.append(df)
                    # print(f"Loaded data from {csv_path}, shape: {df.shape}")
                except Exception as e:
                    print(f"Failed to load data from {csv_path}: {e}")
                    
            print("Data validation successful.")

                
            step_result.data = df_tickers
            
            # for i in range(len(df_tickers)):
            #     print(step_result.data[i].index)
            
            
            return step_result
    
    def __validate_data(self, df: DataFrame) -> bool:
        
        df = df.copy()
        # Implement data validation logic here
        required_columns = {"open", "high", "low", "close", "volume", "ticker"}
        
        if df.empty:
            print("Data validation failed. DataFrame is empty.")
            return False
        if not required_columns.issubset(df.columns):
            print("Data validation failed. Missing columns:", required_columns - set(df.columns))
            return False
        
        if df.isnull().any().any():
            print("Data validation failed. Data contains null values.")
            return False
        
        df["high_is_max"] = df["high"] == df[["open", "high", "low", "close"]].max(axis=1)
        if not df["high_is_max"].all():
            print("Data validation failed. 'high' column does not contain the maximum values.")
            return False
        df = df.drop(columns=["high_is_max"])
        
        df["low_is_min"] = df["low"] == df[["open", "high", "low", "close"]].min(axis=1)
        if not df["low_is_min"].all():
            print("Data validation failed. 'low' column does not contain the minimum values.")
            return False
        df = df.drop(columns=["low_is_min"])

        return True