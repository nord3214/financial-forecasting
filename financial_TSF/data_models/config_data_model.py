from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List


@dataclass
class DataConfig:
    concatenated_csv: str
    number_of_tickers: int
    start_date: str
    end_date: str
    processed_csv: str
    use_cached_processed_data: bool
    
@dataclass
class ModelConfig:
    model_type: str


@dataclass
class FeaturesConfig:
    feature_config_path: str


@dataclass
class PipelineConfig:
    steps: List[str]
    
@dataclass
class ResultsConfig:
    results_path: str
    
@dataclass
class SettingsConfig:
    forecast_horizon: List[int]


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    features: FeaturesConfig
    pipeline: PipelineConfig
    results: ResultsConfig
    settings: SettingsConfig
    
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AppConfig":
        return cls(
            data=DataConfig(
                concatenated_csv=payload["data"]["concatenated_csv"],
                number_of_tickers=payload["data"]["number_of_tickers"],
                start_date=payload["data"]["start_date"],
                end_date=payload["data"]["end_date"],
                processed_csv=payload["data"]["processed_csv"],
                use_cached_processed_data=payload["data"]["use_cached_processed_data"],
            ),
            model=ModelConfig(
                model_type=payload["model"]["modelType"],
            ),
            features=FeaturesConfig(
                feature_config_path=payload["features"]["featureConfigPath"],
            ),

            pipeline=PipelineConfig(
                steps=list(payload["pipeline"]["steps"]),
            ),
            results=ResultsConfig(
                results_path=payload["results"]["resultsPath"],
            ),
            settings=SettingsConfig(
                forecast_horizon=payload["settings"]["forecast_horizon"],
            )
        )