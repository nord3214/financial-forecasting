from financial_TSF.pipeline.pipeline import Pipeline
from financial_TSF.data_models.step_datamodel import StepResult
from datetime import datetime
import torch

# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Number of GPUs:", torch.cuda.device_count())
#     print("Current GPU index:", torch.cuda.current_device())
#     print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
#     print("CUDA version:", torch.version.cuda)


print("Starting pipeline...")
    
start_time: datetime = datetime.now()

pipeline = Pipeline()
print(pipeline.steps)
pipeline.run(step_result=StepResult(start_time=datetime.now()))



