from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd

from financial_TSF.visualization.feature_visualization import FeatureVisualization

class ScalingStep(PipelineStep):
    def __init__(self, config) -> None:
        self.config = config
        self.use_winsorizing = config.pipeline.use_winsorizing
        self.use_scaling = config.pipeline.use_scaling
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
                # x_scaler.fit(pd.concat([split_data['X_train'], split_data['X_val']]))
                # y_scaler.fit(pd.concat([split_data['y_train'], split_data['y_val']]))
                
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
                # add index from X because it also has the ticker in index
                step_result.split_data[idx]['y_train'] = pd.DataFrame(y_train_raw, columns=y_cols, index=split_data['X_train'].index)
                step_result.split_data[idx]['y_val'] = pd.DataFrame(y_val_raw, columns=y_cols, index=split_data['X_val'].index)
                step_result.split_data[idx]['y_test'] = pd.DataFrame(y_test_raw, columns=y_cols, index=split_data['X_test'].index)
                
                step_result.split_data[idx]['x_scaler'] = x_scaler
                step_result.split_data[idx]['y_scaler'] = y_scaler
                # self.show_statistics(step_result.split_data[idx]['X_train'], step_result.split_data[idx]['X_val'], step_result.split_data[idx]['X_test'])
                
                # ----- VALIDATION LOGGING: Index alignment and column preservation -----
                ticker_name = split_data.get("ticker_name", f"Ticker_{idx}")
                print(f"\n[ScalingStep] Validation for '{ticker_name}':")
                
                # Check X columns preserved
                if set(split_data['X_train'].columns) != set(step_result.split_data[idx]['X_train'].columns):
                    print(f"  ERROR: X_train columns changed!")
                    print(f"    Before: {sorted(split_data['X_train'].columns.tolist())}")
                    print(f"    After: {sorted(step_result.split_data[idx]['X_train'].columns.tolist())}")
                else:
                    print(f"  OK: X column order preserved ({len(x_cols)} features)")
                
                # Check y columns preserved
                if set(split_data['y_train'].columns) != set(step_result.split_data[idx]['y_train'].columns):
                    print(f"  ERROR: y_train columns changed!")
                else:
                    print(f"  OK: y column order preserved ({len(y_cols)} targets)")
                
                # Check index alignment by length
                x_train_len = len(step_result.split_data[idx]['X_train'])
                y_train_len = len(step_result.split_data[idx]['y_train'])
                x_val_len = len(step_result.split_data[idx]['X_val'])
                y_val_len = len(step_result.split_data[idx]['y_val'])
                x_test_len = len(step_result.split_data[idx]['X_test'])
                y_test_len = len(step_result.split_data[idx]['y_test'])
                
                print(f"  Index structure (may differ):")
                print(f"    X_train: {type(step_result.split_data[idx]['X_train'].index).__name__}")
                print(f"    y_train: {type(step_result.split_data[idx]['y_train'].index).__name__}")
                
                if x_train_len != y_train_len:
                    print(f"  ERROR: X_train length ({x_train_len}) != y_train length ({y_train_len})")
                else:
                    print(f"  OK: X_train and y_train lengths match ({x_train_len})")
                    
                if x_val_len != y_val_len:
                    print(f"  ERROR: X_val length ({x_val_len}) != y_val length ({y_val_len})")
                else:
                    print(f"  OK: X_val and y_val lengths match ({x_val_len})")
                    
                if x_test_len != y_test_len:
                    print(f"  ERROR: X_test length ({x_test_len}) != y_test length ({y_test_len})")
                else:
                    print(f"  OK: X_test and y_test lengths match ({x_test_len})")
            
            
        x_train_scaled = [split['X_train'] for split in step_result.split_data]
        x_val_scaled = [split['X_val'] for split in step_result.split_data]
        x_test_scaled = [split['X_test'] for split in step_result.split_data]

        
        # if self.use_winsorizing:
        #     visualizer = FeatureVisualization(x_train_scaled, folder=f'/boxplots/Test_Train_robustscaled_winsorized_{self.winsorize_quantile_X*100}_percentiles')
        #     visualizer.run()
        #     visualizer = FeatureVisualization(x_val_scaled, folder=f'/boxplots/Test_Val_robustscaled_winsorized_{self.winsorize_quantile_X*100}_percentiles')
        #     visualizer.run()
        # else:
        #     visualizer = FeatureVisualization(x_train_scaled, folder='/boxplots/robustscaled')
        #     visualizer.run()
        
        # ---- save scaled features for all dataframes for thesis ---- 
        # y_train_scaled = [split['y_train'] for split in step_result.split_data]
        # visualizer = FeatureVisualization(y_train_scaled, folder='results/datasets/targets_scaled_y_train')
        # visualizer.run()
        # y_val_scaled = [split['y_val'] for split in step_result.split_data]
        # visualizer = FeatureVisualization(y_val_scaled, folder='results/datasets/targets_scaled_y_val')
        # visualizer.run()
        
        # visualizer = FeatureVisualization(x_train_scaled, folder='results/datasets/features_scaled_X_train')
        # visualizer.run()
        # visualizer = FeatureVisualization(x_val_scaled, folder='results/datasets/features_scaled_X_val')
        # visualizer.run()
        
        # ---- PRINTing summary statistics after scaling for all dataframes ----
        
        # x_train_scaled = pd.DataFrame(pd.concat(x_train_scaled, axis=0))
        # x_val_scaled = pd.DataFrame(pd.concat(x_val_scaled, axis=0))
        # x_test_scaled = pd.DataFrame(pd.concat(x_test_scaled, axis=0))
        # y_train_scaled = pd.DataFrame(pd.concat(y_train_scaled, axis=0))
        # y_val_scaled = pd.DataFrame(pd.concat(y_val_scaled, axis=0))
        
        # print("Overall scaled X_train_features: ", x_train_scaled.describe().round(4))
        # print("Overall scaled X_val_features: ", x_val_scaled.describe().round(4))
        # print("Overall scaled X_test_features: ", x_test_scaled.describe().round(4)
        #       )
        # print("Overall scaled y_train_targets: ", y_train_scaled.describe().round(4))
        # print("Overall scaled y_val_targets: ", y_val_scaled.describe().round(4))
        
            
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
    

    def show_statistics(self, X_train, X_val, X_test):
        """Print summary statistics for the corresponding sets and selected features"""
        feature_columns = X_train.columns  
        feature_columns = feature_columns[:5]
        feature_columns = ['LOG_VOLUME', 'LOGRET_1', 'Kurtosis_50' ]
        statistics = []       
        for dataset in [X_train, X_val, X_test]:
            print("Statitsics for ticker: ", dataset.index.get_level_values('ticker')[0])
            for feature in feature_columns:
                mean = dataset[feature].mean()
                std = dataset[feature].std()
                median = dataset[feature].median()
                set_name = "Training" if dataset.equals(X_train) else "Validation" if dataset.equals(X_val) else "Test"
                statistics.append({
                    'Feature': feature,
                    'Dataset': set_name,
                    'Mean': mean,
                    'Std': std,
                    'Median': median
                })
        statistics_df = pd.DataFrame(statistics)
        statistics_df.set_index(['Feature'], inplace=True)
        statistics_df = statistics_df.sort_index()
        statistics_df.set_index('Dataset', append=True, inplace=True)
        
        latex_code = statistics_df.to_latex(
            index=True, 
            multicolumn=True, 
            multicolumn_format='c', 
            bold_rows=False,
        )

        print(statistics_df.round(2))
        print(latex_code)