from pandas import DataFrame
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd

from financial_TSF.visualization.feature_visualization import FeatureVisualization


class ScalingStep(PipelineStep):
    def __init__(self, config) -> None:
        self.config = config


    def execute(self, step_result: StepResult) -> StepResult:
        data = step_result.data 
        print("Received data for Normalizing Step with", len(data), "dataframes.")
        #need to normalize for deep learning models left out for now for random forest
        
        # visualizer = FeatureVisualization(data)
        # visualizer.run()
        
        
        for i, df in enumerate(data):
            print(f"Scaling DataFrame {i+1} with shape: {df.shape}")
            
            df_non_numeric = df.select_dtypes(exclude=['float64', 'int64'])
            data[i] = self.perform_min_max_scaling(df)  
            data[i] = pd.concat([data[i], df_non_numeric], axis=1) 
            
            data[i] = df         
            
            print(f"Scaled DataFrame {i+1}, new shape: {data[i].shape}")
        step_result.data = data
        return step_result
    
    def perform_standard_scaling(self, df: DataFrame) -> DataFrame:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # :TODO handle non-numeric columns seperately
        df = df.select_dtypes(include=['float64', 'int64'])
        train_size = int((1-0.2) * len(df))
        df_train = df.iloc[:train_size]

        scaler.fit(df_train)
        df = DataFrame(scaler.transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.columns, index=df.index)
        # print("train df: ", df[:train_size].describe().round(2))
        # print("test df: ", df[train_size:].describe().round(2))
        # print("Overall df: ", df.describe().round(2))
        
        return df
    
    def perform_rolling_standard_scaling(self, df: DataFrame, window_size: int = 100) -> DataFrame:
        
        # :TODO handle non-numeric columns seperately
        df = df.select_dtypes(include=['float64', 'int64'])
        
        mean = df.rolling(window=window_size).mean()
        std = df.rolling(window=window_size).std()
        
        df_scaled = (df - mean) / std
        print(df_scaled.describe().round(2))
        # df_scaled = DataFrame(scaled_array, columns=df.columns, index=df.index)
        
        return df_scaled
    
    def perform_expanding_standard_scaling(self, df: DataFrame) -> DataFrame:
        # :TODO handle non-numeric columns seperately
        df = df.select_dtypes(include=['float64', 'int64'])
        
        mean = df.expanding().mean()
        std = df.expanding().std()
        
        df_scaled = (df - mean) / std
        print(df_scaled.describe().round(2))
        # df_scaled = DataFrame(scaled_array, columns=df.columns, index=df.index)
        
        return df_scaled
    
    
    def perform_min_max_scaling(self, df: DataFrame) -> DataFrame:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        # :TODO handle non-numeric columns seperately
        df = df.select_dtypes(include=['float64', 'int64'])
        train_size = int((1-0.2) * len(df))
        df_train = df.iloc[:train_size]

        scaler.fit(df_train)
        df = DataFrame(scaler.transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.columns, index=df.index)
        print("train df: ", df[:train_size].describe().round(2))
        print("test df: ", df[train_size:].describe().round(2))
        print("Overall df: ", df.describe().round(2))
        return df
    
    def perform_rolling_min_max_scaling(self, df: DataFrame, window_size: int = 100) -> DataFrame:
        
        # :TODO handle non-numeric columns seperately
        df = df.select_dtypes(include=['float64', 'int64'])
        
        min_val = df.rolling(window=window_size).min()
        max_val = df.rolling(window=window_size).max()
        
        df_scaled = (df - min_val) / (max_val - min_val)
        print(df_scaled.describe().round(2))
        # df_scaled = DataFrame(scaled_array, columns=df.columns, index=df.index)
        
        return df_scaled