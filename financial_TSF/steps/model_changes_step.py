from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.steps.base_step import PipelineStep


class ModelChangesStep(PipelineStep):
    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.model_type: str = config.model.model_type

    def execute(self, step_result):
        if self.model_type == "PatchTST":
            print("Using PatchTST model, applying specific pretraining steps.")
            
            # -------- rename columns for PatchTST --------
            X_train, X_val, X_test = step_result.X_train, step_result.X_val, step_result.X_test
            for df in [X_train, X_val, X_test]:
                df['ds'] = df.index.get_level_values('timestamp')
                df['unique_id'] = df.index.get_level_values('ticker')
            
            y_train, y_val, y_test = step_result.y_train, step_result.y_val, step_result.y_test
            for df in [y_train, y_val, y_test]:
                df.rename(columns={df.columns[0]: 'y'}, inplace=True)
            
            step_result.X_train = X_train
            step_result.y_train = y_train
            step_result.X_val = X_val
            step_result.y_val = y_val
            step_result.X_test = X_test
            step_result.y_test = y_test
            
            return step_result
            
        else: 
            print(f"No seperate model changes step needed for {self.model_type} model, skipping.")
            
            return step_result
