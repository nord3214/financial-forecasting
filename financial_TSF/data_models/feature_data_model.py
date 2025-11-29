from dataclasses import dataclass
from typing import List, Mapping, Any


@dataclass
class TrendIndicatorsConfig:
    enabled: bool
    indicators: List[str]


@dataclass
class TechnicalIndicatorsConfig:
    enabled: bool
    trend_indicators: TrendIndicatorsConfig


@dataclass
class TimeFeaturesConfig:
    enabled: bool
    features: List[str]


@dataclass
class FeaturesConfig:
    technical_indicators: TechnicalIndicatorsConfig
    time_features: TimeFeaturesConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeaturesConfig":
        block = payload["features"]
        return cls(
            technical_indicators=TechnicalIndicatorsConfig(
                enabled=block["technical_indicators"]["enabled"],
                trend_indicators=TrendIndicatorsConfig(
                    enabled=block["technical_indicators"]["trend_indicators"]["enabled"],
                    indicators=list(block["technical_indicators"]["trend_indicators"]["indicators"]),
                ),
            ),
            time_features=TimeFeaturesConfig(
                enabled=block["time_features"]["enabled"],
                features=list(block["time_features"]["features"]),
            ),
        )