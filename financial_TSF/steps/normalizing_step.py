from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd

from financial_TSF.visualization.feature_visualization import FeatureVisualization

class ScalingStep(PipelineStep):
    def __init__(self, config) -> None:
        self.config = config
        self.scalers = []
        self.use_winsorizing = True


    def execute(self, step_result: StepResult) -> StepResult:
        """
        Scales data using the corresponding scaler
        !!! Also removes ticker column for now !!!
        """
        print("Received data for Normalizing Step with", len(step_result.split_data), "dataframes.")
        # :TODO add multiindex with ticker!!=!=
        
        if self.use_winsorizing:
            
            for split in step_result.split_data:
                df_train = split['X_train']
                df_test = split['X_test']
                df_train = df_train.set_index('ticker', append=True)
                df_test = df_test.set_index('ticker', append=True)
                lower = df_train.quantile(0.01)
                upper = df_train.quantile(0.99)
                df_train = df_train.clip(lower=lower, upper=upper, axis=1)
                df_test = df_test.clip(lower=lower, upper=upper, axis=1)
                split['X_test'] = df_test
                split['X_train'] = df_train
                
        
        # visualizer = FeatureVisualization(x_train, folder='30_unscaled_features_clipped_1percentiles')
        # visualizer.run()
        
        for idx, split_data in enumerate(step_result.split_data):
            
           
            x_scaler = self.get_scaler()
            y_scaler = self.get_scaler()
            
            x_scaler.fit(split_data['X_train'])
            y_scaler.fit(split_data['y_train'])
            
            # Extract column names before transforming
            x_cols = split_data['X_train'].columns
            y_cols = split_data['y_train'].columns

            xt_raw = x_scaler.transform(split_data['X_train'])
            xv_raw = x_scaler.transform(split_data['X_test'])
            yt_raw = y_scaler.transform(split_data['y_train'])
            yv_raw = y_scaler.transform(split_data['y_test'])
            

            # Convert back to DataFrames with original Index and Column names
            step_result.split_data[idx]['X_train'] = pd.DataFrame(xt_raw, columns=x_cols, index=split_data['X_train'].index)
            step_result.split_data[idx]['X_test'] = pd.DataFrame(xv_raw, columns=x_cols, index=split_data['X_test'].index)
            step_result.split_data[idx]['y_train'] = pd.DataFrame(yt_raw, columns=y_cols, index=split_data['X_train'].index)
            step_result.split_data[idx]['y_test'] = pd.DataFrame(yv_raw, columns=y_cols, index=split_data['X_test'].index)
            
            # y_test_multiindex = step_result.split_data[idx]['y_test'].copy()
            # y_test_multiindex['ticker'] = step_result.split_data[idx]['ticker_name']
            # y_test_multiindex = y_test_multiindex.set_index('ticker', append=True)
            # import numpy as np

            # # Vergleicht nur die nackten Zahlenwerte beider DataFrames
            # try:
            #     np.testing.assert_allclose(y_test_multiindex.values, step_result.split_data[idx]['y_test'].values, rtol=1e-5)
            #     print("Check bestanden: Die numerischen Werte sind identisch.")
            # except AssertionError as e:
            #     print(f"Abweichung gefunden: {e}")
            
            step_result.split_data[idx]['x_scaler'] = x_scaler
            step_result.split_data[idx]['y_scaler'] = y_scaler
            
            # print(f"Scaled X_train: ", step_result.split_data[idx]['X_train'].describe().round(5))
            # print(f"Scaled y_train: ", step_result.split_data[idx]['y_train'].describe().round(5))
            # print(f"Scaled X_test: ", step_result.split_data[idx]['X_test'].describe().round(5))
            # print(f"Scaled y_test: ", step_result.split_data[idx]['y_test'].describe().round(5)) 
            
        # x_train_scaled = [split['X_train'] for split in step_result.split_data]
        # visualizer = FeatureVisualization(x_train_scaled, folder='30_scaled_features_standard_scaler_clipped_1percentiles')
        # visualizer.run()
        
        # y_train_scaled = [split['y_train'] for split in step_result.split_data]
        # visualizer = FeatureVisualization(y_train_scaled, folder='30_scaled_features_standard_scaler_clipped_1percentiles')
        # visualizer.run()
            
            
        # :TODO put all together for training step again :)
        # for backwards compatibility done here
        step_result.X_train = pd.concat([step_result.split_data[i]['X_train'] for i in range(len(step_result.split_data))], axis=0)
        step_result.X_test = pd.concat([step_result.split_data[i]['X_test'] for i in range(len(step_result.split_data))], axis=0)
        step_result.y_train = pd.concat([step_result.split_data[i]['y_train'] for i in range(len(step_result.split_data))], axis=0)
        step_result.y_test = pd.concat([step_result.split_data[i]['y_test'] for i in range(len(step_result.split_data))], axis=0)
        
        # ---------- sort for time order -----------
        # step_result.X_train = step_result.X_train.sort_index(kind='stable')
        # step_result.X_test = step_result.X_test.sort_index(kind='stable')
        # step_result.y_train = step_result.y_train.sort_index(kind='stable')
        # step_result.y_test = step_result.y_test.sort_index(kind='stable')
        
        # check if order is correct
        
        # print(step_result.X_train.shape, step_result.y_train.shape, step_result.X_test.shape, step_result.y_test.shape)
        # print("Described X_train: ", step_result.X_train.describe().round(5))
            
        
        return step_result
    
    def get_scaler(self):
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        # scaler = RobustScaler()
        return scaler
    
    def perform_standard_scaling(self, df: DataFrame) -> DataFrame:
        scaler = StandardScaler()
        # :TODO handle non-numeric columns seperately
        df = df.select_dtypes(include=['float64', 'int64'])
        train_size = int((1-0.2) * len(df))
        df_train = df.iloc[:train_size]

        scaler.fit(df_train)
        df = DataFrame(scaler.transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.columns, index=df.index)
        
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