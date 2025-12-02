from pandas import DataFrame
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.steps.base_step import PipelineStep


class NormalizingStep(PipelineStep):
    def __init__(self, config) -> None:
        self.config = config


    def execute(self, step_result: StepResult) -> StepResult:
        data = step_result.data 
        print("Received data for Normalizing Step with", len(data), "dataframes.")
        #need to normalize for deep learning models left out for now for random forest
        for i, df in enumerate(data):
            print(f"Normalizing DataFrame {i+1} with shape: {df.shape}")
            
            print(df.describe())
            
            print(f"Normalized DataFrame {i+1}, new shape: {data[i].shape}")
        step_result.data = data
        return step_result