from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd

from financial_TSF.visualization.feature_visualization import FeatureVisualization

class ScalingStep(PipelineStep):
    def __init__(self, config) -> None:
        self.config = config
        self.use_winsorizing = True
        self.use_scaling = True
        self.winsorize_quantile_X = 0.01
        self.winsorize_quantile_y = 0.005


    def execute(self, step_result: StepResult) -> StepResult:
        """
        Scales data using the corresponding scaler
        !!! Also removes ticker column for now !!!
        """
        print("Received data for Normalizing Step with", len(step_result.split_data), "dataframes.")
        
        # create unscaled boxplots for all dataframes before scaling
        # x_train_scaled = [split['X_train'] for split in step_result.split_data]
        # visualizer = FeatureVisualization(x_train_scaled, folder='boxplots/unscaled')
        # visualizer.run()
        
        if self.use_winsorizing:
            
            for split in step_result.split_data:
                # winsorizing for feature columns
                df_train = split['X_train']
                df_val = split['X_val']
                df_test = split['X_test']
                
                df_train = df_train.set_index('ticker', append=True)
                df_val = df_val.set_index('ticker', append=True)
                df_test = df_test.set_index('ticker', append=True)
                
                lower_x = df_train.quantile(self.winsorize_quantile_X)
                upper_x = df_train.quantile(1 - self.winsorize_quantile_X)
                
                df_train = df_train.clip(lower=lower_x, upper=upper_x, axis=1)
                df_val = df_val.clip(lower=lower_x, upper=upper_x, axis=1)
                df_test = df_test.clip(lower=lower_x, upper=upper_x, axis=1)
                
                split['X_test'] = df_test
                split['X_val'] = df_val
                split['X_train'] = df_train
                
                # winsorizing for target columns
                df_train = split['y_train']
                df_val = split['y_val']
                df_test = split['y_test']

                lower_y = df_train.quantile(self.winsorize_quantile_y)            
                upper_y = df_train.quantile(1 - self.winsorize_quantile_y)
                
                df_train = df_train.clip(lower=lower_y, upper=upper_y, axis=1)
                df_val = df_val.clip(lower=lower_y, upper=upper_y, axis=1)
                df_test = df_test.clip(lower=lower_y, upper=upper_y, axis=1)
                
                split['y_test'] = df_test
                split['y_val'] = df_val
                split['y_train'] = df_train
                
        # create boxplots after winsorizing but before scaling
        # x_train_scaled = [split['X_train'] for split in step_result.split_data]
        # visualizer = FeatureVisualization(x_train_scaled, folder=f'boxplots/winsorized_{self.winsorize_quantile_X*100}_percentiles')
        # visualizer.run()
        
        for idx, split_data in enumerate(step_result.split_data):
            if not self.use_winsorizing:
                split_data['X_train'].set_index('ticker', append=True, inplace=True)
                split_data['X_val'].set_index('ticker', append=True, inplace=True)
                split_data['X_test'].set_index('ticker', append=True, inplace=True)
                
            if self.use_scaling:
            
                x_scaler = self.get_scaler()
                y_scaler = self.get_scaler()
                
                x_scaler.fit(split_data['X_train'])
                y_scaler.fit(split_data['y_train'])
                
                # Extract column names before transforming
                x_cols = split_data['X_train'].columns
                y_cols = split_data['y_train'].columns

                x_train_raw = x_scaler.transform(split_data['X_train'])
                x_val_raw = x_scaler.transform(split_data['X_val'])
                x_test_raw = x_scaler.transform(split_data['X_test'])
                y_train_raw = y_scaler.transform(split_data['y_train'])
                y_val_raw = y_scaler.transform(split_data['y_val'])
                y_test_raw = y_scaler.transform(split_data['y_test'])
                
                # Convert back to DataFrames with original Index and Column names
                step_result.split_data[idx]['X_train'] = pd.DataFrame(x_train_raw, columns=x_cols, index=split_data['X_train'].index)
                step_result.split_data[idx]['X_val'] = pd.DataFrame(x_val_raw, columns=x_cols, index=split_data['X_val'].index)
                step_result.split_data[idx]['X_test'] = pd.DataFrame(x_test_raw, columns=x_cols, index=split_data['X_test'].index)
                step_result.split_data[idx]['y_train'] = pd.DataFrame(y_train_raw, columns=y_cols, index=split_data['X_train'].index)
                step_result.split_data[idx]['y_val'] = pd.DataFrame(y_val_raw, columns=y_cols, index=split_data['X_val'].index)
                step_result.split_data[idx]['y_test'] = pd.DataFrame(y_test_raw, columns=y_cols, index=split_data['X_test'].index)
                
                step_result.split_data[idx]['x_scaler'] = x_scaler
                step_result.split_data[idx]['y_scaler'] = y_scaler
            
        x_train_scaled = [split['X_train'] for split in step_result.split_data]
        
        # if self.use_winsorizing:
        #     visualizer = FeatureVisualization(x_train_scaled, folder=f'/boxplots/robustscaled_winsorized_{self.winsorize_quantile_X*100}_percentiles')
        #     visualizer.run()
        # else:
        #     visualizer = FeatureVisualization(x_train_scaled, folder='/boxplots/robustscaled')
        #     visualizer.run()
        
        # y_train_scaled = [split['y_train'] for split in step_result.split_data]
        # visualizer = FeatureVisualization(y_train_scaled, folder='targets_scaled_robust_scaler_winsorized0_5percentiles')
        # visualizer.run()
            
            
        # put all together for training step again :)
        step_result.X_train = pd.concat([step_result.split_data[i]['X_train'] for i in range(len(step_result.split_data))], axis=0)
        step_result.X_val = pd.concat([step_result.split_data[i]['X_val'] for i in range(len(step_result.split_data))], axis=0)
        step_result.X_test = pd.concat([step_result.split_data[i]['X_test'] for i in range(len(step_result.split_data))], axis=0)
        step_result.y_train = pd.concat([step_result.split_data[i]['y_train'] for i in range(len(step_result.split_data))], axis=0)
        step_result.y_val = pd.concat([step_result.split_data[i]['y_val'] for i in range(len(step_result.split_data))], axis=0)
        step_result.y_test = pd.concat([step_result.split_data[i]['y_test'] for i in range(len(step_result.split_data))], axis=0)
        
        #  ---------- sort for time order -----------
        # sort multiindex by level 0 (timestamp)
        
        # step_result.X_train.sort_index(level=0, inplace=True)
        # step_result.X_val.sort_index(level=0, inplace=True)
        # step_result.X_test.sort_index(level=0, inplace=True)
        # step_result.y_train.sort_index(level=0, inplace=True)   
        # step_result.y_val.sort_index(level=0, inplace=True)
        # step_result.y_test.sort_index(level=0, inplace=True)
        
        return step_result
    
    def get_scaler(self):
        # scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaler = RobustScaler()
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