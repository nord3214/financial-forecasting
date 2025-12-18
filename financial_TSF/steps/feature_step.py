import json
from pathlib import Path
from typing import Any, Dict

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
        data = step_result.data
        print("Received data for FeatureStep with", len(data), "dataframes.")

        # Go over all dataframes(different Tickers) and apply technical indicators
        for i, df in enumerate(data):
            print(f"Processing DataFrame {i+1} with shape: {df.shape}")
            
            # Go over all technical indicators from config and apply them
            
            tech_indicators = TechnicalIndicators(self.config.features.feature_config_path, df)
            
            data[i] = tech_indicators.calculate_indicators()
            print(f"Processed DataFrame {i+1}, new shape: {data[i].shape}")
            
            self.__save_processed_data(data[i])
            
        columns = data[0].columns.tolist()
        print("Columns after technical indicators: ", columns)

                        
        step_result.data = data
        return step_result
            
    
    
    
    
    
    
    def __save_processed_data(self, data: DataFrame) -> None:
        # data = data.round(10)
        print("Dataframe: ", data.tail(30))
        ticker = data["ticker"].iloc[0]
        print("Saving processed data for ticker:", ticker)
        
        save_path = f'{self.config.data.processed_csv}/{ticker}/{ticker}_intraday_5min_processed.csv'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False)
        print(f"Processed data saved to {save_path}")
       
        
    
    
    
