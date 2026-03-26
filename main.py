import json

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

feature_combinations = [
    [10],
    [20],
    [30],
    [10, 20],
    [10, 30],
    [20, 30],
    [10, 20, 30]
    
]
for combination in feature_combinations:
    
        
    print("Starting pipeline with features:", combination)
    
    # handle writing complexity levels to config file
    with open("financial_TSF/config.json", "r") as f:
        config = json.load(f)
        
    config["features"]["complexity_levels"] = combination

    with open("financial_TSF/config.json", "w") as f:
        json.dump(config, f, indent=4)

    start_time: datetime = datetime.now()

    pipeline = Pipeline()
    
    print(pipeline.steps)
    pipeline.run(step_result=StepResult(start_time=datetime.now()))
    



