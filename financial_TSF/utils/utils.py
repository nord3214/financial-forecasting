from datetime import datetime
from pathlib import Path

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult


def get_result_path(config: AppConfig, step_result: StepResult)-> Path:
    
    save_path = Path(config.results.results_path)
    
    timestamp = step_result.start_time.strftime("%H%M%S_%d%m%Y") 
    feature_complexity_str = "_".join(map(str, config.features.complexity_levels))
    
    run_dir: Path = save_path / config.model.model_type / feature_complexity_str / f"{timestamp}_{config.model.model_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir