import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from pandas import DataFrame
from financial_TSF.data_models.config_data_model import AppConfig, FeaturesConfig
from financial_TSF.features.technical_indicator import TechnicalIndicators
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep


class FeatureStep(PipelineStep):
    
    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
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
                
                # Go over all technical indicators from config and apply them
                
                tech_indicators = TechnicalIndicators(self.config.features.feature_config_path, df)
                
                data[i] = tech_indicators.calculate_indicators()
                data[i] = data[i].round(10)
                print(f"Processed DataFrame {i+1}, new shape: {data[i].shape}")

                with pd.option_context('display.max_columns', None):
                    print(data[i].describe())
                
                self.__save_processed_data(data[i])
                
            columns = data[0].columns.tolist()
            print("Columns after technical indicators: ", columns)

                            
            step_result.data = data
            print(f"New calculated features with {[df.shape for df in data]} shapes.")
            
            
            
            print("Loading processed data from cache...")
            df_loaded = []
            for ticker in step_result.ticker_names:
                processed_df = self.__load_processed_data(ticker)
                df_loaded.append(processed_df)
            
            df1 = data[0]
            df2 = df_loaded[0]
            
            print(df1.shape.__eq__(df2.shape))
            print(df1.columns.equals(df2.columns))
            print(df1.index.equals(df2.index))
            print(df1.dtypes.compare(df2.dtypes))
            print(df1.compare(df2))
            print(df1.equals(df2))



            
            return step_result

            return step_result
            
    
    #
    
    
    
    
    
    def __save_processed_data(self, data: DataFrame) -> None:
        # data = data.round(10)
        print("Dataframe: ", data.tail(30))
        ticker = data["ticker"].iloc[0]
        print("Saving processed data for ticker:", ticker)
        
        save_path = f'{self.config.data.processed_csv}/{ticker}/{ticker}_intraday_5min_processed.csv'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False, float_format='%.15g')
        print(f"Processed data saved to {save_path}")
        
    def __load_processed_data(self, ticker: str) -> DataFrame:
        load_path = f'{self.config.data.processed_csv}/{ticker}/{ticker}_intraday_5min_processed.csv'
        print("Loading processed data for ticker:", ticker)
        data = pd.read_csv(load_path)
        print(f"Processed data loaded from {load_path}, shape: {data.shape}")
        return data
       
        
    
    
    
