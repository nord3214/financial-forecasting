import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from pandas import DataFrame
from financial_TSF.data_models.config_data_model import AppConfig, FeaturesConfig
from financial_TSF.features.technical_indicator import TechnicalIndicators
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep


class FeatureStep(PipelineStep):
    
    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.complexity_levels = self.config.features.complexity_levels
        
        with open(self.config.features.feature_config_path, "r") as f:
            self.feature_config: Dict[str, Any] = json.load(f)
        
            
    
    
    def execute(self, step_result: StepResult) -> StepResult:
        
        if self.config.data.use_cached_processed_data:
            print("Loading processed data from cache...")
            data = []
            for ticker in step_result.ticker_names:
                processed_df = self.__load_processed_data(ticker)
                data.append(processed_df)
            step_result.data = data
            
            print(f"Loaded precalculated features with {[df.shape for df in data]} shapes.")
            
            return step_result
        
        else:
            data = step_result.data
            print("Received data for FeatureStep with", len(data), "dataframes.")

            # Go over all dataframes(different Tickers) and apply technical indicators
            for i, df in enumerate(data):
                print(f"Processing DataFrame {i+1} with shape: {df.shape}")
                
                current_ticker = df["ticker"].iloc[0]
                
                # also calculate log return of current interval of ticker (for economic evaluation)
                log_return_horizon1 = pd.DataFrame(index=df.index)
                log_return_horizon1['actual_log_return'] = np.log(df['close'] / df['close'].shift(1))
                log_return_horizon1['return_next_interval'] = log_return_horizon1['actual_log_return'].shift(-1)
                
                step_result.split_data.append({
                    "ticker_name": current_ticker,
                    "X_train": None,
                    "X_test": None, 
                    "y_train": None,
                    "y_train_pred": None,
                    "y_test": None,
                    "y_test_pred": None,
                    "x_scaler": None,
                    "y_scaler": None,             
                    "log_return_horizon1": log_return_horizon1['return_next_interval'],
                })
                
                
                # Go over all technical indicators from config and apply them
                tech_indicators = TechnicalIndicators(self.config.features.feature_config_path, df)
                
                data[i] = tech_indicators.calculate_indicators(self.complexity_levels)
                

                data[i] = data[i].round(10)
                print(f"Processed DataFrame {i+1}, new shape: {data[i].shape}")

                with pd.option_context('display.max_columns', None):
                    print(data[i].describe())
                
                self.__save_processed_data(current_ticker, data[i])
                
            columns = data[0].columns.tolist()
            print("Columns after technical indicators: ", columns)

            step_result.data = data
            print(f"New calculated features with {[df.shape for df in data]} shapes.")
            
            # for df in data:
            #     print("Index is", df.index)
            
            
            return step_result
            
    
    #
    
    
    
    
    
    def __save_processed_data(self, ticker: str, data: DataFrame) -> None:
        # data = data.round(10)
        print("Dataframe: ", data.tail(30))
        print("Saving processed data for ticker:", ticker)
        
        save_path = f'{self.config.data.processed_csv}/{ticker}/{ticker}_intraday_5min_processed.parquet'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(save_path, index=False)
        # data.to_csv(save_path, index=False, float_format='%.15g')
        print(f"Processed data saved to {save_path}")
        
    def __load_processed_data(self, ticker: str) -> DataFrame:
        load_path = f'{self.config.data.processed_csv}/{ticker}/{ticker}_intraday_5min_processed.parquet'
        print("Loading processed data for ticker:", ticker)
        # data = pd.read_csv(load_path)
        data = pd.read_parquet(load_path)
        print(f"Processed data loaded from {load_path}, shape: {data.shape}")
        return data
       
        
    
    
    
