import json
from typing import Any
from pandas import DataFrame

from financial_TSF.data_models.feature_data_model import FeaturesConfig





class TechnicalIndicators:
    def __init__(self, config_path: str, data: DataFrame):
        with open(config_path, "r") as f:
            self.config: FeaturesConfig = FeaturesConfig.from_dict(json.load(f))
        
        self.data: DataFrame = data
        
    def calculate_indicators(self, data: DataFrame) -> DataFrame:
        """Calculate technical indicators based on the configuration."""
        for indicator in self.config.technical_indicators.trend_indicators.indicators:
            print(f"Calculating {indicator}...")
            if indicator == "SMA":
                data = self.simple_moving_average(data, window_size=20)

            
        return data
        
    # def parameter_wrapper(self, func, params: list[Any]):
    #     for param in params:
    #         func(param)


    def simple_moving_average(self, data: DataFrame, window_size: int) -> DataFrame:
        """Calculate the moving average for Close price."""
        data[f'MA_{window_size}'] = data['close'].rolling(window=window_size).mean()
        
    
        return data