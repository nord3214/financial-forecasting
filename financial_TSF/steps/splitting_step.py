import numpy as np
import pandas as pd
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
from financial_TSF.visualization.financial_eda import FinancialEDA

HIGHLY_CORRELATED_FEATURES: list[str] = ['open', 'high', 'low', 'close']


class SplittingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.forecast_horizons = config.settings.forecast_horizon
        self.train_ratio = 0.75
        self.val_ratio = 0.15
        self.highly_correlated_features = HIGHLY_CORRELATED_FEATURES

        
    def execute(self, step_result: StepResult) -> StepResult:

        target_column_names = []
        max_number_of_rows = 0
        # initialize split points as timestamps
        val_start_timestamp = None
        test_start_timestamp = None
        train_start_timestamp = None
        test_end_timestamp = None
        
        step_result.X_train = []
        step_result.X_val = []
        step_result.X_test = [] 
        
        step_result.y_train = []
        step_result.y_val = []
        step_result.y_test = []
        
        # loop1 determin global split points based on largest dataframe
        for i, ticker_df in enumerate(step_result.data):
            step_result.data[i], target_column_names = self.calculate_target_columns(ticker_df)
            
            # remove highly correlated features if any (correlation > 0.xx)
            step_result.data[i] = step_result.data[i].drop(columns=self.highly_correlated_features)
                        
            # remove rows with NaN values created by shifting for target columns/ or feature calculation
            step_result.data[i] = step_result.data[i].dropna()
            
            # get split point based on the largest dataframe
            if step_result.data[i].shape[0] > max_number_of_rows:
                max_number_of_rows = step_result.data[i].shape[0]
                
                val_start_idx = int(max_number_of_rows * self.train_ratio)
                val_start_timestamp = step_result.data[i].index[val_start_idx]
                
                test_start_idx = int(max_number_of_rows * (self.train_ratio + self.val_ratio))  
                test_start_timestamp = step_result.data[i].index[test_start_idx]
                train_start_timestamp = step_result.data[i].index[0]
                test_end_timestamp = step_result.data[i].index[-1]
                
        # Safety Check: Found split timestamps
        if val_start_timestamp is None or test_start_timestamp is None:
            raise ValueError("No valid data found to determine split points.")
                
        print("Global split points determined:")
        print(f"Train set: {train_start_timestamp} to {val_start_timestamp}")
        print(f"Val set: {val_start_timestamp} to {test_start_timestamp}")
        print(f"Test set: {test_start_timestamp} to {test_end_timestamp}")

        #list for tickername to remove due to insufficient data - remove after loop
        tickers_to_remove = []        
        
        # loop2 split each ticker df based on global split points        
        for i, ticker_df in enumerate(step_result.data):
            
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(ticker_df, val_start_timestamp=val_start_timestamp, test_start_timestamp=test_start_timestamp, target_column_names=target_column_names)

            ticker_name = ticker_df["ticker"].iloc[0]
            if X_train.empty or X_val.empty:
                print(f"Not enough data for ticker {ticker_name} to split. Wont use this ticker for training/testing.")
                tickers_to_remove.append(ticker_name)
                    # step_result.ticker_names.remove(ticker_name)
                # also remove corresponding split_data entry of this ticker
                    # step_result.split_data = [data for data in step_result.split_data if data["ticker_name"] != ticker_name]
                continue
    
            # append to step result all train/test data
            step_result.X_train.append(X_train)
            step_result.X_val.append(X_val)
            step_result.X_test.append(X_test)
            step_result.y_train.append(y_train)
            step_result.y_val.append(y_val)
            step_result.y_test.append(y_test)
            
            # Update split_data dictionary            
            for data in step_result.split_data:
                if data["ticker_name"] == ticker_name:
                    data["X_train"] = X_train
                    data["X_val"] = X_val
                    data["X_test"] = X_test
                    data["y_train"] = y_train
                    data["y_val"] = y_val
                    data["y_test"] = y_test
                    break
        
        # eda = FinancialEDA(None, output_path="plots/eda/splitting_step/")
        # eda.plot_split_points(step_result.data, val_start_timestamp, test_start_timestamp)
                
        # remove tickers with insufficient data from step_result        
        if tickers_to_remove:
            step_result.ticker_names = [t for t in step_result.ticker_names if t not in tickers_to_remove]
            step_result.split_data = [data for data in step_result.split_data if data["ticker_name"] not in tickers_to_remove]
            
            
        # clear step_result data to save memory and avoid confustion because no more needed
        step_result.data = []
        
        
        return step_result
    
    def calculate_target_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # save all target column names to return
        target_column_names = []
        for horizon in self.forecast_horizons:
            column_name = f"target_{horizon}"
            
            df_temp = df.copy()
            df_temp["LOGRET_1"] = np.log(df_temp['close'] / df_temp['close'].shift(1))
            df[column_name] = df_temp['LOGRET_1'].rolling(window=horizon).sum().shift(-horizon)
            target_column_names.append(column_name)
                
        return df, target_column_names
    
    def split_data(self, df: pd.DataFrame, val_start_timestamp, test_start_timestamp, target_column_names: list[str] = []) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Docstring for split_data
        
        :param df: df of ticker to split
        :type df: pd.DataFrame
        :param val_start_timestamp: validation start timestamp
        :type val_start_timestamp: pd.Timestamp
        :param test_start_timestamp: test start timestamp
        :type test_start_timestamp: pd.Timestamp
        :param target_column_names: name of target columns in y
        :type target_column_names: list[str]
        :return: X_train, X_val, X_test, y_train, y_val, y_test
        :rtype: tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]
        """
        
        df = df.sort_index()
        
        y = df[target_column_names]
        X = df.drop(columns=target_column_names)

        
        train_mask = df.index < val_start_timestamp
        val_mask = (df.index >= val_start_timestamp) & (df.index < test_start_timestamp)
        test_mask = df.index >= test_start_timestamp
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        return X_train, X_val, X_test, y_train, y_val, y_test

        
