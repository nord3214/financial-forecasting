import numpy as np
import pandas as pd
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep


class SplittingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.forecast_horizons = config.settings.forecast_horizon
        self.train_size = 0.8
        
    def execute(self, step_result: StepResult) -> StepResult:

        target_column_names = []
        max_number_of_rows = 0
        split_point_timestamp = 0
        
        step_result.X_train = []
        step_result.X_test = [] 
        step_result.y_train = []
        step_result.y_test = []
        
        for i, ticker_df in enumerate(step_result.data):
            step_result.data[i], target_column_names = self.calculate_target_columns(ticker_df)
            
            # remove non-stationary features here 
            step_result.data[i] = step_result.data[i].drop(['open', 'high', 'low', 'close'], axis=1)
            
            # remove rows with NaN values created by shifting for target columns/ or feature calculation
            step_result.data[i] = step_result.data[i].dropna()
            
            if step_result.data[i].shape[0] > max_number_of_rows:
                max_number_of_rows = step_result.data[i].shape[0]
                split_point_idx = int(max_number_of_rows * self.train_size)
                split_point_timestamp = step_result.data[i].index[split_point_idx]
                
                
        
                
        for i, ticker_df in enumerate(step_result.data):
            
            X_train, X_test, y_train, y_test = self.split_data(ticker_df, split_point_timestamp=split_point_timestamp, target_column_names=target_column_names)

            ticker_name = ticker_df["ticker"].iloc[0]
            if X_train.empty:
                print(f"Not enough data for ticker {ticker_name} to split. Wont use this ticker for training/testing.")
                step_result.ticker_names.remove(ticker_name)
                # also remove corresponding split_data entry of this ticker
                step_result.split_data = [data for data in step_result.split_data if data["ticker_name"] != ticker_name]
                continue
            
            step_result.X_train.append(X_train)
            step_result.X_test.append(X_test)
            step_result.y_train.append(y_train)
            step_result.y_test.append(y_test)
            
            
            for data in step_result.split_data:
                if data["ticker_name"] == ticker_name:
                    data["X_train"] = X_train
                    data["X_test"] = X_test
                    data["y_train"] = y_train
                    data["y_test"] = y_test
            
        # clear step_result data to save memory and avoid confustion because no more needed
        step_result.data = []
        
        
        return step_result
    
    def calculate_target_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # save all target column names to return
        target_column_names = []
            # :TODO handle target creation earlier in the pipeline maybe in feature step
        for horizon in self.forecast_horizons:
            column_name = f"target_{horizon}"
            
            df_temp = df.copy()
            df_temp["LOGRET_1"] = np.log(df_temp['close'] / df_temp['close'].shift(1))
            df[column_name] = df_temp['LOGRET_1'].rolling(window=horizon).sum().shift(-horizon)
            target_column_names.append(column_name)
                
        return df, target_column_names
    
    def split_data(self, df: pd.DataFrame, split_point_timestamp, train_size: float = 0.8, target_column_names: list[str] = []) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Docstring for split_data
        
        :param df: df of ticker to split
        :type df: pd.DataFrame
        :param train_size: train size ratio between 0 and 1
        :type train_size: float
        :param target_column_names: name of target columns in y
        :type target_column_names: list[str]
        :param split_point_timestamp: timestamp to split the data on
        :type split_point_timestamp: pd.Timestamp
        :return: X_train, X_test, y_train, y_test
        :rtype: tuple[DataFrame, DataFrame, DataFrame, DataFrame]
        """
        
        df = df.sort_index()
        
        y = df[target_column_names]
        X = df.drop(columns=target_column_names)

        
        train_mask = df.index < split_point_timestamp
        test_mask = df.index >= split_point_timestamp
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        return X_train, X_test, y_train, y_test

        
