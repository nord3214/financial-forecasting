import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.features.technical_indicator import TechnicalIndicators
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
from financial_TSF.visualization.financial_eda import FinancialEDA


HIGHLY_CORRELATED_FEATURES: list[str] = ['open', 'high', 'low', 'close']


class FeatureStep(PipelineStep):
    
    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.complexity_levels = self.config.features.complexity_levels
        
        with open(self.config.features.feature_config_path, "r") as f:
            self.feature_config: Dict[str, Any] = json.load(f)
            
        self.highly_correlated_features = HIGHLY_CORRELATED_FEATURES
        
            
    
    
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
                # print(f"Processing DataFrame {i+1} with shape: {df.shape}")
                
                current_ticker = df["ticker"].iloc[0]
                
                # also calculate log return of current interval of ticker (for economic evaluation)
                log_return_horizon1 = pd.DataFrame(index=df.index)
                log_return_horizon1['actual_log_return'] = np.log(df['close'] / df['close'].shift(1))
                log_return_horizon1['return_next_interval'] = log_return_horizon1['actual_log_return'].shift(-1)
                log_return_horizon1['ticker'] = current_ticker
                log_return_horizon1 = log_return_horizon1.set_index('ticker', append=True)
                
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
                    "log_return_horizon1": log_return_horizon1[['return_next_interval']],
                })
                
                
                # Go over all technical indicators from config and apply them
                tech_indicators = TechnicalIndicators(self.config.features.feature_config_path, df)
                
                data[i] = tech_indicators.calculate_indicators(self.complexity_levels)
                

                data[i] = data[i].round(10)
                
                # remove highly correlated features
                data[i] = data[i].drop(columns=self.highly_correlated_features)
                
                # print(f"Processed DataFrame {i+1}, new shape: {data[i].shape}")

                # with pd.option_context('display.max_columns', None):
                #     print(data[i].describe())
                
                self.__save_processed_data(current_ticker, data[i])
                
            columns = data[0].columns.tolist()
            print("Columns after technical indicators: ", columns)
            
            # run_eda(data[0], step_result.ticker_names[0])  # Just use the first ticker for EDA
            get_highly_correlated_features(data[0], threshold=0.9)
            
            return step_result
    
    def __save_processed_data(self, ticker: str, data: DataFrame) -> None:
        # data = data.round(10)
        # print("Dataframe: ", data.tail(30))
        # print("Saving processed data for ticker:", ticker)
        
        save_path = f'{self.config.data.processed_csv}/{ticker}/{ticker}_intraday_5min_processed.parquet'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(save_path, index=False)
        # data.to_csv(save_path, index=False, float_format='%.15g')
        # print(f"Processed data saved to {save_path}")
        
    def __load_processed_data(self, ticker: str) -> DataFrame:
        load_path = f'{self.config.data.processed_csv}/{ticker}/{ticker}_intraday_5min_processed.parquet'
        # print("Loading processed data for ticker:", ticker)
        # data = pd.read_csv(load_path)
        data = pd.read_parquet(load_path)
        # print(f"Processed data loaded from {load_path}, shape: {data.shape}")
        return data
    


def run_eda(df: DataFrame, ticker: str) -> None:
    eda = FinancialEDA(df)  # Just use the first ticker for EDA
    # ----------- EDA on OHLCV ----------------
    eda.plot_time_series(column='close', title=f'Close Price of {ticker}')
    
    eda.check_adfuller('close')
    eda.check_adfuller('LOGRET_1')
    eda.plot_distribution(column='LOGRET_1', range=(-0.025, 0.025))
    eda.plot_distribution(column='volume', range=(0, 1.5e7))
    eda.plot_distribution(column='LOG_VOLUME')
    
    # ----------- EDA on calculated features ----------------
    
    eda.plot_time_series(column='EMA_20', title=f'EMA 20 of {ticker}')
    
    for col in eda.df.columns:
        if col in ['SMA_20', 'EMA_20', '%BBANDS', 'OBV', 'DAILY_CLOSE' ]:
            eda.plot_time_series(column=col, title=f'{col} of {ticker}')

    
    for feature_col in eda.df.columns:
        if feature_col not in ['ticker']:
            p_value = eda.check_adfuller(feature_col)
            if p_value >= 0.05:
                print(f"Feature {feature_col} is non-stationary (p-value: {p_value})!!!!")
                    
def get_highly_correlated_features(df: DataFrame, threshold: float = 0.95) -> list[str]:
    
    feature_cols = [col for col in df.columns if col not in ['ticker']]
    corr_matrix = df[feature_cols].corr()
    
    # only use lower triangle of correlation matrix to not double count
    mask = np.tril(np.ones(corr_matrix.shape)).astype(bool)
    reduced_matrix = corr_matrix.where(mask)

    corr_pairs: pd.Series = pd.Series(reduced_matrix.unstack())
    sorted_pairs= corr_pairs.sort_values()

    # filter to get high corr and not self-correlation
    high_corr = sorted_pairs[(abs(sorted_pairs) > threshold) & (sorted_pairs != 1.0)]

    print("Highly correlated feature pairs (>0.95 or <-0.95):")
    if high_corr.empty:
        print("No pairs found above threshold.")
    else:
        print(f"Found {len(high_corr)} highly correlated feature pairs:")
        print(high_corr)
        
    return high_corr.index.get_level_values(0).tolist()

        
    
    
    
