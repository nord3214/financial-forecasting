
from financial_TSF.steps.data_loading_step import DataLoadingStep
from financial_TSF.steps.feature_step import FeatureStep


STEP_REGISTRY = {
    "FeatureStep": FeatureStep,
    "DataLoadingStep": DataLoadingStep
}