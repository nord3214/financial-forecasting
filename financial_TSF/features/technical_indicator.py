import json
from typing import Any
from numpy import float64
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import talib

from financial_TSF import data
from financial_TSF.data_models.feature_data_model import FeaturesConfig





class TechnicalIndicators:
    def __init__(self, config_path: str, data: DataFrame):
        with open(config_path, "r") as f:
            self.config: dict[str, Any] = (json.load(f))
        
        self.data: DataFrame = data
        
            
        self.feature_registry = {
        "SMA": self.simple_moving_average,
        "EMA": self.exponential_moving_average,
        "RSI": self.relative_strength_index,
        "ROC": self.rate_of_change,
        "ADX": self.average_directional_movement_index,
        "DMI": self.directional_movement_index,
        "BBANDS": self.bollinger_bands,
        "NATR": self.normalized_average_true_range,
        "OBV": self.on_balance_volume,
        "LOGRET": self.log_return,
        "VI": self.vortex_indicator,
        "STOCH": self.stochastic_oscillator,
        "CCI": self.commodity_channel_index,
        "MI": self.mass_index,
        "Kurtosis": self.kurtosis,
        "PVO": self.volume_oscillator,
        "MACD": self.moving_average_convergence_divergence,
        "MFI": self.money_flow_index,
        "DAY_OF_WEEK": self.day_of_week,
        "MINUTE_OF_DAY": self.min_of_day,
        "DAILY_CLOSE": self.daily_closing_price,
        "VOL_SPIKE": self.volume_spike,
        "VOL_RATIO": self.volume_ratio,
        "VOLATILITY_RETURN": self.volatility_return,
        "HL_RANGE": self.hl_range,
        "OC_RANGE": self.oc_range,
        "TURNOVER": self.turnover,
        "LOG_VOLUME": self.log_volume,
    }

    def apply_indicator(self, feature_name: str, params: list | dict | int):
        """Dynamically apply a technical indicator based on its name and parameters."""
            
        if feature_name not in self.feature_registry:
            raise ValueError(f"Indicator '{feature_name}' not recognized.")
        
        func = self.feature_registry[feature_name]
        
        # handle None params 
        if not params:
            self.data = func(feature_name, self.data)
        
        # handle int as param
        elif isinstance(params, int):
            self.data =func(feature_name, self.data, params)
        
        # handle list(int) of params 
        elif isinstance(params, list) and all(isinstance(p, int) for p in params):
            for p in params:
                self.data = func(feature_name, self.data, p)
                
        #handle dict of params 
        elif isinstance(params, dict):
            self.data = func(feature_name, self.data, **params)
            
        # handle list of dicts
        elif isinstance(params, list) and all(isinstance(p, dict) for p in params):
            for p in params:
                self.data = func(feature_name, self.data, **p)
                
                
        else:
            raise ValueError(f"Parameters: '{params}' for indicator '{feature_name}' are not in a recognized format.")
        
                
    def calculate_indicators(self, complexity_levels: list[int]) -> DataFrame:

        # :TODO create model class for config
        # :TODO do scaling / normalization somewhere
        for name, params in self.config['technical_indicators'].items():
            # print(f"Calculating {name} with params: {params}...")
            if params['enabled'] == True:
                if params.get('complexity', None) in complexity_levels:
                    self.apply_indicator(name, params.get('params', None))
        if 10 not in complexity_levels:
            self.data = self.remove_basic_columns_from_dataset(self.data)
            
        print("Final data after technical indicators:\n", self.data.info())
            
        return self.data
    
    def remove_basic_columns_from_dataset(self, data: DataFrame) -> DataFrame:
        """Remove basic columns from the dataset, keeping only technical indicators."""
        # :TODO somewhere remove index timestamp
        basic_columns = ['open', 'high', 'low', 'close', 'volume', 'ticker']
        
        return data.drop(columns=basic_columns)
                
    # def calculate_indicators_old(self, data: DataFrame) -> DataFrame:
    #     """Calculate technical indicators based on the configuration."""
    #     for indicator in self.config  .technical_indicators.trend_indicators.indicators:
            
    #         print(f"Calculating {indicator}...")
    #         if indicator == "SMA":
    #         # :TODO need to use features from config
    #             data = self.simple_moving_average(data, window_size=20)
    #             # data = self.simple_moving_average_ta_lib(data, window_size=20)
                
    #             data = self.simple_moving_average(data, window_size=50)
    #             # data = self.simple_moving_average_ta_lib(data, window_size=50)
    #             data = self.simple_moving_average(data, window_size=200)
    #             # data = self.simple_moving_average_ta_lib(data, window_size=200)
                
    #             data = self.exponential_moving_average(data, window_size=20)
    #             data = self.exponential_moving_average(data, window_size=50)    
    #             data = self.exponential_moving_average(data, window_size=200)
                
    #             data = self.average_directional_movement_index(data, window_size=14)

    #             # data = self.moving_average_convergence_divergence(data)
    #             data = self.directional_movement_index(data, window_size=14)
    #             data = self.relative_strength_index(data, window_size=14)
    #             data = self.williams_percent_range(data, window_size=14)
    #             data = self.rate_of_change(data, window_size=10)
    #             data = self.bollinger_bands(data, window_size=20, num_std_dev=2)
    #             data = self.normalized_average_true_range(data, window_size=14)
    #             data = self.on_balance_volume(data)
                
    #             data = self.log_return(data)
                
                
                
                
                
        
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
        



    def simple_moving_average(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the moving average for Close price."""
        data[f'{feature_name}_{window_size}'] = data['close'].rolling(window=window_size).mean()
        
        return data
    
    def simple_moving_average_ta_lib(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the moving average for Close price using TA-Lib."""
        data[f'{feature_name}_{window_size}_ta_lib'] = talib.SMA(data['close'], timeperiod=window_size)
    
        return data
    
    def exponential_moving_average(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Exponential Moving Average (EMA)."""
        data[f'{feature_name}_{window_size}'] = talib.EMA(data['close'], timeperiod=window_size)
        
        return data
    
    def average_directional_movement_index(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Average Directional Movement Index (ADX)."""
        data[f'{feature_name}_{window_size}'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def moving_average_convergence_divergence(self, feature_name, data, fast_period: int, slow_period: int, signal_period: int):
        """Calculate the Moving Average Convergence Divergence (MACD)."""
        macd, macdsignal, macdhist = talib.MACD(data['close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        data[f'{feature_name}_MACD'] = macd
        data[f'{feature_name}_MACD_Signal'] = macdsignal
        data[f'{feature_name}_MACD_Hist'] = macdhist
        
        return data
        
    def directional_movement_index(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Directional Movement Index (DMI)."""  
        data[f'{feature_name}_{window_size}'] = talib.DX(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def relative_strength_index(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Relative Strength Index (RSI)."""
        data[f'{feature_name}_{window_size}'] = talib.RSI(data['close'], timeperiod=window_size)
        
        return data
    
    def williams_percent_range(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Williams %R."""
        data[f'{feature_name}_{window_size}'] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def rate_of_change(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Rate of Change (ROC)."""
        data[f'{feature_name}_{window_size}'] = talib.ROC(data['close'], timeperiod=window_size)
        
        return data
    
    def bollinger_bands(self, feature_name, data, window_size: int, num_std_dev: float) -> DataFrame:
        """Calculate the Bollinger Bands."""
        upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=window_size, nbdevup=num_std_dev, nbdevdn=num_std_dev)
        data[f'{feature_name}_Upper_{window_size}'] = upperband
        data[f'{feature_name}_Middle_{window_size}'] = middleband
        data[f'{feature_name}_Lower_{window_size}'] = lowerband
        
        return data
    
    def normalized_average_true_range(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Normalized Average True Range (NATR)."""
        data[f'{feature_name}_{window_size}'] = talib.NATR(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def on_balance_volume(self, feature_name, data) -> DataFrame:
        """Calculate the On-Balance Volume (OBV)."""
        data[f'{feature_name}'] = talib.OBV(data['close'], data['volume'])
        
        return data
    
    def log_return(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Log Return."""
        data[f'{feature_name}_{window_size}'] = np.log(data['close'] / data['close'].shift(window_size))
        return data
    
    def volume_spike(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate Volume Spike as the ratio of current volume to average volume over a window."""
        data[f'{feature_name}_{window_size}'] = data['volume'] / data['volume'].rolling(window=window_size).mean()
        return data
    
    def volume_ratio(self, feature_name, data, fast_period: int, slow_period: int) -> DataFrame:
        """Calculate Volume Ratio of n periods as the ratio of current volume to volume m periods ago."""
        data[f'{feature_name}_{fast_period}_{slow_period}'] = data['volume'].rolling(fast_period).mean() / data['volume'].rolling(slow_period).mean()
        return data
    
    def volatility_return(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate Volatility Return as the standard deviation of log returns over a window."""
        df = data.copy()
        df = self.log_return("log_return", df, 1)
        data[f'{feature_name}_{window_size}'] = df[f'log_return_1'].rolling(window=window_size).std()
        
        return data
    
    def hl_range(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the High-Low Range over a rolling window."""
        data[f'{feature_name}_{window_size}'] = (data['high'] - data['low']).rolling(window=window_size).mean()
        
        return data
    
    # :TODO add hl range kind of ratio over n periods
    
    def oc_range(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Open-Close Range over a rolling window."""
        data[f'{feature_name}_{window_size}'] = (data['open'] - data['close']).rolling(window=window_size).mean()
        
        return data
    
    def daily_closing_price(self, feature_name, data) -> DataFrame:
        """Extract the daily closing price."""
        df = data.copy()
        df['date'] = df.index.date
        daily_close = df.groupby(df['date'])['close'].last()
        daily_close_prev = daily_close.shift(1)
        data = data.merge(daily_close_prev.rename(f'{feature_name}'), left_on=df.index.date, right_on=daily_close_prev.index, how='left')
        data.drop(columns=['key_0'], inplace=True)
        # print(data[[f'{feature_name}', 'close']].tail(200))
        
        return data
    
    def turnover(self, feature_name, data) -> DataFrame:
        """Calculate Turnover as volume multiplied by close price."""
        data[f'{feature_name}'] = data['volume'] * data['close']
        
        return data
    
    def vortex_indicator(self, feature_name, data, window_size) -> DataFrame:
        """Calculate the Vortex Indicator (VI)"""
        df = data.copy()
        df['TR'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['VM_plus'] = abs(df['high'] - df['low'].shift(1))
        df['VM_minus'] = abs(df['low'] - df['high'].shift(1))
        
        df['TR_sum'] = df['TR'].rolling(window=window_size).sum()
        df['VM_plus_sum'] = df['VM_plus'].rolling(window=window_size).sum()
        df['VM_minus_sum'] = df['VM_minus'].rolling(window=window_size).sum()
        
        data[f'{feature_name}_plus_{window_size}'] = df['VM_plus_sum'] / df['TR_sum']
        data[f'{feature_name}_minus_{window_size}'] = df['VM_minus_sum'] / df['TR_sum']
        
        data[f'{feature_name}_diff_{window_size}'] = (data[f'{feature_name}_plus_{window_size}'] - data[f'{feature_name}_minus_{window_size}'])
        
        # :TODO adjust how to represent the indicator
        
        return data
    
    def stochastic_oscillator(self, feature_name, data, window_size: int, smooth_k: int, smooth_d: int) -> DataFrame:
        """Calculate the Stochastic Oscillator."""
        slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'], 
                                   fastk_period=window_size, 
                                   slowk_period=smooth_k, 
                                   slowd_period=smooth_d, 
                                   )
        data[f'{feature_name}_%K_{window_size}'] = slowk
        data[f'{feature_name}_%D_{window_size}'] = slowd
        
        return data
    
    def commodity_channel_index(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Commodity Channel Index (CCI)."""
        data[f'{feature_name}_{window_size}'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=window_size)
        
        return data
    
    def mass_index(self, feature_name, data: pd.DataFrame, ema_period: int = 9, sum_period: int = 25) -> pd.DataFrame:
        """Calculate the Mass Index (MI) – a volatility indicator"""

        df = data.copy()

        df['Range'] = df['high'] - df['low']
        
        df['EMA1'] = df['Range'].ewm(span=ema_period, adjust=False).mean()
        df['EMA2'] = df['EMA1'].ewm(span=ema_period, adjust=False).mean()

        df['Ratio'] = df['EMA1'] / df['EMA2']
        df['MassIndex'] = df['Ratio'].rolling(window=sum_period).sum()
        data[f'{feature_name}_{ema_period}_{sum_period}'] = df[['MassIndex']]

        return data
    
    def kurtosis(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Kurtosis over a rolling window."""
        df = data.copy()
        df = self.log_return("log_return", df, 1)
        data[f'{feature_name}_{window_size}'] = df['log_return_1'].rolling(window=window_size).kurt()
        
        print(data[[f'{feature_name}_{window_size}']].describe())
        return data
    
    def volume_oscillator(self, feature_name, data, fast_period: int, long_period: int, signal_period: int) -> DataFrame:
        """Calculate the Percentage Volume Oscillator (PVO)."""
        df = data.copy()
        df['VMA_short'] = df['volume'].rolling(window=fast_period).mean()
        df['VMA_long'] = df['volume'].rolling(window=long_period).mean()
        df['PVO'] = ((df['VMA_short'] - df['VMA_long']) / df['VMA_long']) * 100
        df['PVO_signal'] = df['PVO'].rolling(window=signal_period).mean()
        df['PVO_hist'] = df['PVO'] - df['PVO_signal']
        
        
        data[[f'{feature_name}', f'{feature_name}_signal', f'{feature_name}_hist']] = df[['PVO', 'PVO_signal', 'PVO_hist']]
        return data
    
    def money_flow_index(self, feature_name, data, window_size: int) -> DataFrame:
        """Calculate the Money Flow Index (MFI)."""
        data[f'{feature_name}_{window_size}'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=window_size)
        
        return data
    
    def day_of_week(self, feature_name, data) -> DataFrame:
        """Extract the day of the week from the timestamp."""
        df = data.copy()
        df['Day_of_Week'] = df.index.dayofweek  # Monday=0, Sunday=6
        df['Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 6)
        df['Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 6)
        # data['Day_of_Week_sin'] = df['Day_of_Week_sin']
        # data['Day_of_Week_cos'] = df['Day_of_Week_cos']
        data[f'{feature_name}'] = df['Day_of_Week']
        return data 
    
    def min_of_day(self, feature_name, data) -> DataFrame:
        """Extract the minute of the day from the timestamp."""
        df = data.copy()
        df['Minute_of_Day'] = df.index.hour * 60 + df.index.minute
        df['Minute_of_Day_sin'] = np.sin(2 * np.pi * df['Minute_of_Day'] / 1440)
        df['Minute_of_Day_cos'] = np.cos(2 * np.pi * df['Minute_of_Day'] / 1440)
        # data['Minute_of_Day_sin'] = df['Minute_of_Day_sin']
        # data['Minute_of_Day_cos'] = df['Minute_of_Day_cos']
        data[f'{feature_name}'] = df['Minute_of_Day']
        return data
    
    def log_volume(self, feature_name, data) -> DataFrame:
        """Calculate the Log Volume."""
        data[f'{feature_name}'] = np.log(data['volume']+1)  # Adding 1 to avoid log(0)
        
        return data
    
    
    
    
    
    

