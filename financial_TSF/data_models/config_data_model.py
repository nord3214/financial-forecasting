from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List


@dataclass
class DataConfig:
    concatenated_csv: str
    use_all_tickers: bool
    processed_csv: str
    
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
                use_all_tickers=payload["data"]["useAllTickers"],
                processed_csv=payload["data"]["processed_csv"],
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