from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.steps.base_step import PipelineStep


class DataLoadingStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.csv_path = config.data.concatenated_csv
        self.use_all_tickers = config.data.use_all_tickers

    def execute(self, data = None, model = None):
        # Implement data loading logic here
        print("Executing Data Loading Step with csv_path:", self.csv_path, "and use_all_tickers:", self.use_all_tickers)
        return data