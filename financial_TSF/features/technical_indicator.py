import json
from typing import Any
from numpy import float64
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import talib

from financial_TSF.data_models.feature_data_model import FeaturesConfig





class TechnicalIndicators:
    def __init__(self, config_path: str, data: DataFrame):
        with open(config_path, "r") as f:
            self.config = (json.load(f))
        
        self.data: DataFrame = data
        
            
        self.indicator_registry = {
        "SMA": self.simple_moving_average,
        "EMA": self.exponential_moving_average,
        "RSI": self.relative_strength_index,
        "ROC": self.rate_of_change,
        "ADX": self.average_directional_movement_index,
        "DMI": self.directional_movement_index,
        "BBANDS": self.bollinger_bands,
        "NATR": self.normalized_average_true_range,
        "OBV": self.on_balance_volume,
        "LOGRET": self.log_return
    }

    def apply_indicator(self, indicator: str, params: list | dict | int):
        """Dynamically apply a technical indicator based on its name and parameters."""
        
        if indicator not in self.indicator_registry:
            raise ValueError(f"Indicator '{indicator}' not recognized.")
        
        func = self.indicator_registry[indicator]
        
        # handle bool params
        if isinstance(params, bool):
            if params:
                self.data = func(self.data)
        
        # handle int as param
        elif isinstance(params, int):
            self.data =func(self.data, params)
        
        # handle list(int) of params 
        elif isinstance(params, list) and all(isinstance(p, int) for p in params):
            for p in params:
                self.data = func(self.data, p)
                
        #handle dict of params 
        elif isinstance(params, dict):
            self.data = func(self.data, **params)
            
        # handle list of dicts
        elif isinstance(params, list) and all(isinstance(p, dict) for p in params):
            for p in params:
                self.data = func(self.data, **p)
                
                
        else:
            raise ValueError(f"Parameters: '{params}' for indicator '{indicator}' are not in a recognized format.")
        
                
    def calculate_indicators(self) -> DataFrame:
        # :TODO create model class for config
        for name, params in self.config['technical_indicators'].items():
            print(f"Calculating {name} with params: {params}...")
            self.apply_indicator(name, params)
            
        return self.data
                
    def calculate_indicators_old(self, data: DataFrame) -> DataFrame:
        """Calculate technical indicators based on the configuration."""
        for indicator in self.config.technical_indicators.trend_indicators.indicators:
            
            print(f"Calculating {indicator}...")
            if indicator == "SMA":
            # :TODO need to use features from config
                data = self.simple_moving_average(data, window_size=20)
                # data = self.simple_moving_average_ta_lib(data, window_size=20)
                
                data = self.simple_moving_average(data, window_size=50)
                # data = self.simple_moving_average_ta_lib(data, window_size=50)
                data = self.simple_moving_average(data, window_size=200)
                # data = self.simple_moving_average_ta_lib(data, window_size=200)
                
                data = self.exponential_moving_average(data, window_size=20)
                data = self.exponential_moving_average(data, window_size=50)    
                data = self.exponential_moving_average(data, window_size=200)
                
                data = self.average_directional_movement_index(data, window_size=14)

                # data = self.moving_average_convergence_divergence(data)
                data = self.directional_movement_index(data, window_size=14)
                data = self.relative_strength_index(data, window_size=14)
                data = self.williams_percent_range(data, window_size=14)
                data = self.rate_of_change(data, window_size=10)
                data = self.bollinger_bands(data, window_size=20, num_std_dev=2)
                data = self.normalized_average_true_range(data, window_size=14)
                data = self.on_balance_volume(data)
                
                data = self.log_return(data)
                
                
                
                
                
        
        # data['timestamp'] = pd.to_datetime(data['timestamp'])
        # data.set_index('timestamp', inplace=True)
        
        # Agregating to daily data
        # data = data.resample('D').agg({
        #     "open": "first",
        #     "high": "max",
        #     "low": "min",
        #     "close": "last",
        #     "volume": "sum",
        #     "ticker": "first"
        # })

        data.info()
        print(data.tail())
        
        
        # daily aggregated data
        
        print(data["Log_Return"].describe())
            
        return data
        



    def simple_moving_average(self, data, window_size: int) -> DataFrame:
        """Calculate the moving average for Close price."""
        data[f'MA_{window_size}'] = data['close'].rolling(window=window_size).mean()
        
        return data
    
    def simple_moving_average_ta_lib(self, data, window_size: int) -> DataFrame:
        """Calculate the moving average for Close price using TA-Lib."""
        data[f'MA_{window_size}_ta_lib'] = talib.SMA(data['close'], timeperiod=window_size)
    
        return data
    
    def exponential_moving_average(self, data, window_size: int) -> DataFrame:
        """Calculate the Exponential Moving Average (EMA)."""
        data[f'EMA_{window_size}'] = talib.EMA(data['close'], timeperiod=window_size)
        
        return data
    
    def average_directional_movement_index(self, data, window_size: int) -> DataFrame:
        """Calculate the Average Directional Movement Index (ADX)."""
        data[f'ADX_{window_size}'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def moving_average_convergence_divergence(self, data):
        """Calculate the Moving Average Convergence Divergence (MACD)."""
        macd, macdsignal, macdhist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_Signal'] = macdsignal
        data['MACD_Hist'] = macdhist
        
    def directional_movement_index(self, data, window_size: int) -> DataFrame:
        """Calculate the Directional Movement Index (DMI)."""  
        data[f'DMI_{window_size}'] = talib.DX(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def relative_strength_index(self, data, window_size: int) -> DataFrame:
        """Calculate the Relative Strength Index (RSI)."""
        data[f'RSI_{window_size}'] = talib.RSI(data['close'], timeperiod=window_size)
        
        return data
    
    def williams_percent_range(self, data, window_size: int) -> DataFrame:
        """Calculate the Williams %R."""
        data[f'WILLR_{window_size}'] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def rate_of_change(self, data, window_size: int) -> DataFrame:
        """Calculate the Rate of Change (ROC)."""
        data[f'ROC_{window_size}'] = talib.ROC(data['close'], timeperiod=window_size)
        
        return data
    
    def bollinger_bands(self, data, window_size: int, num_std_dev: float) -> DataFrame:
        """Calculate the Bollinger Bands."""
        upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=window_size, nbdevup=num_std_dev, nbdevdn=num_std_dev)
        data[f'BB_Upper_{window_size}'] = upperband
        data[f'BB_Middle_{window_size}'] = middleband
        data[f'BB_Lower_{window_size}'] = lowerband
        
        return data
    
    def normalized_average_true_range(self, data, window_size: int) -> DataFrame:
        """Calculate the Normalized Average True Range (NATR)."""
        data[f'NATR_{window_size}'] = talib.NATR(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def on_balance_volume(self, data) -> DataFrame:
        """Calculate the On-Balance Volume (OBV)."""
        data['OBV'] = talib.OBV(data['close'], data['volume'])
        
        return data
    
    def log_return(self, data) -> DataFrame:
        """Calculate the Log Return."""
        data['Log_Return'] = np.log(data['close'] / data['close'].shift(1))
        
        return data
    
    
    
    

