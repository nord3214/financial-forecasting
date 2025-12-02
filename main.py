from financial_TSF.pipeline.pipeline import Pipeline
from financial_TSF.data_models.step_datamodel import StepResult



print("Starting pipeline...")
pipeline = Pipeline()
print(pipeline.steps)
pipeline.run(step_result=StepResult())

