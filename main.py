from financial_TSF.pipeline.pipeline import Pipeline
from financial_TSF.data_models.step_datamodel import StepResult
from datetime import datetime


print("Starting pipeline...")
    
start_time: datetime = datetime.now()

pipeline = Pipeline()
print(pipeline.steps)
pipeline.run(step_result=StepResult(start_time=datetime.now()))

end_time = datetime.now()
print("Pipeline finished. Duration: ", end_time - start_time)


