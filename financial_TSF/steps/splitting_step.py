import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.rf.random_forest_regressor import SETTINGS
from financial_TSF.steps.base_step import PipelineStep


class SplittingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:

        target_column_names = []
        
        for i, ticker_df in enumerate(step_result.data):
            step_result.data[i], target_column_names = self.calculate_target_columns(ticker_df)
            
            # remove rows with NaN values created by shifting for target columns/ or feature calculation
            step_result.data[i] = step_result.data[i].dropna()
            
        # concatenate all dataframes into one for splitting 
        df = pd.concat(step_result.data, axis = 0)
        print("all dfs together: ", df.head(10))
        print("all dfs together sorter: ", df.sort_index().head(10))
        
        if "ticker" in df.columns:
            df = df.drop("ticker", axis=1)

            
        step_result.X_train, step_result.X_test, step_result.y_train, step_result.y_test = self.split_data(df, target_column_names=target_column_names)
        
        
        # y = df.iloc[:, -len(self.config.settings.forecast_horizon):]
        # X = df.iloc[:, :-len(self.config.settings.forecast_horizon)]
        
        
        # step_result.X_train, step_result.X_test, step_result.y_train, step_result.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        
        return step_result
    
    def calculate_target_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # save all target column names to return
        target_column_names = []
            # :TODO handle target creation earlier in the pipeline maybe in feature step
        for horizon in self.config.settings.forecast_horizon:
            column_name = f"target_{horizon}"
            
            df_temp = df.copy()
            df_temp["LOGRET_1"] = np.log(df_temp['close'] / df_temp['close'].shift(1))
            df[column_name] = df_temp['LOGRET_1'].rolling(window=horizon).sum().shift(-horizon)
            target_column_names.append(column_name)
                
        return df, target_column_names
    
    def split_data(self, df: pd.DataFrame, train_size: float = 0.8, target_column_names: list[str] = []) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        df = df.sort_index()
        
        y = df[target_column_names]
        X = df.drop(columns=target_column_names)

        # 80/20 split based on time
        unique_dates = df.index.unique()
        
        # gives the index to split on
        split_point = unique_dates[int(len(unique_dates) * train_size)]
        
        train_mask = df.index < split_point
        test_mask = df.index >= split_point
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        return X_train, X_test, y_train, y_test

        
