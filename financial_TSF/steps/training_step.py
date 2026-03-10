from pathlib import Path

import joblib

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.MLP.MLP_model import MLPModel
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.base_step import PipelineStep


class TrainingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        self.use_hyperparmeter_tuning = config.pipeline.use_hyperparameter_tuning
        
    def execute(self, step_result: StepResult) -> StepResult:
        
        model_path = None
        #PatchTST load already fitted model ONLY PATH NOT FILE
        # model_path = Path("resultsTraining/PatchTST/10_20_30/155851_10032026_PatchTST/saved_model")
        
        # MLP load already fitted model
        # model_path = Path("resultsTraining/MLP/10_20_30/162812_10032026_MLP/saved_model/model.pt")
        
        # RF load already fitted model
        # model_path = Path("resultsTraining/RF/10_20_30/162115_10032026_RF/random_forest_model.joblib")
        
        
        

        model_instance: BaseModel = step_result.model_instance

        if model_path is None:
            
            if self.use_hyperparmeter_tuning:
                model_instance.tune_hyperparameters(step_result)
            # --------- fit --------        
            model_instance.fit(step_result)
            model_instance.save_model(step_result)
            
        else: 
            print(f"Loading model from {model_path}...")
            path_str = model_path._str
            if (self.config.model.model_type not in path_str):
                raise ValueError(f"Provided model path {model_path} does most likely not match the configured model type {self.config.model.model_type}.")
            model_instance.load_model(model_path)
        
        # -------- predict --------
        step_result.y_test, step_result.y_test_pred, step_result.y_train, step_result.y_train_pred = model_instance.predict(step_result)
            
        print("Training completed.")
        return step_result