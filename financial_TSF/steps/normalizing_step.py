from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd

class ScalingStep(PipelineStep):
    def __init__(self, config) -> None:
        self.config = config
        self.scalers = []


    def execute(self, step_result: StepResult) -> StepResult:
        """
        Scales data using the corresponding scaler
        !!! Also removes ticker column for now !!!
        """
        data = step_result.data 
        print("Received data for Normalizing Step with", len(data), "dataframes.")
        #need to normalize for deep learning models left out for now for random forest
        
        # visualizer = FeatureVisualization(data)
        # visualizer.run()
        
        for idx, split_data in enumerate(step_result.split_data):
            
            # X_ticker = step_result.X_train.loc[step_result.X_train['ticker'] == ticker_name]
            # y_ticker = step_result.y_train.loc[step_result.y_train.index.isin(X_ticker.index)]
            
            
                        
            x_scaler = self.get_scaler()
            y_scaler = self.get_scaler()
            
            x_scaler.fit(split_data['X_train'].drop(columns=['ticker']))
            y_scaler.fit(split_data['y_train'])
            
            # Extract column names before transforming
            x_cols = split_data['X_train'].drop(columns=['ticker']).columns
            y_cols = split_data['y_train'].columns

            xt_raw = x_scaler.transform(split_data['X_train'].drop(columns=['ticker']))
            xv_raw = x_scaler.transform(split_data['X_test'].drop(columns=['ticker']))
            yt_raw = y_scaler.transform(split_data['y_train'])
            yv_raw = y_scaler.transform(split_data['y_test'])
            

            # Convert back to DataFrames with original Index and Column names
            step_result.split_data[idx]['X_train'] = pd.DataFrame(xt_raw, columns=x_cols, index=split_data['X_train'].index)
            step_result.split_data[idx]['X_test'] = pd.DataFrame(xv_raw, columns=x_cols, index=split_data['X_test'].index)
            step_result.split_data[idx]['y_train'] = pd.DataFrame(yt_raw, columns=y_cols, index=split_data['y_train'].index)
            step_result.split_data[idx]['y_test'] = pd.DataFrame(yv_raw, columns=y_cols, index=split_data['y_test'].index)
            
            step_result.split_data[idx]['x_scaler'] = x_scaler
            step_result.split_data[idx]['y_scaler'] = y_scaler
            
            # print(f"Scaled X_train: ", step_result.split_data[idx]['X_train'].describe().round(5))
            # print(f"Scaled y_train: ", step_result.split_data[idx]['y_train'].describe().round(5))
            # print(f"Scaled X_test: ", step_result.split_data[idx]['X_test'].describe().round(5))
            # print(f"Scaled y_test: ", step_result.split_data[idx]['y_test'].describe().round(5)) 
            
            
            # :TODO put all together for training step again :)
        # for backwards compatibility done here
        step_result.X_train = pd.concat([step_result.split_data[i]['X_train'] for i in range(len(step_result.split_data))], axis=0)
        step_result.X_test = pd.concat([step_result.split_data[i]['X_test'] for i in range(len(step_result.split_data))], axis=0)
        step_result.y_train = pd.concat([step_result.split_data[i]['y_train'] for i in range(len(step_result.split_data))], axis=0)
        step_result.y_test = pd.concat([step_result.split_data[i]['y_test'] for i in range(len(step_result.split_data))], axis=0)
        
        # print(step_result.X_train.shape, step_result.y_train.shape, step_result.X_test.shape, step_result.y_test.shape)
        # print("Described X_train: ", step_result.X_train.describe().round(5))
            
        
        return step_result
    
    def get_scaler(self):
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        return scaler
    
    def perform_standard_scaling(self, df: DataFrame) -> DataFrame:
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
    
    
    def perform_min_max_scaling(self, df: DataFrame):
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