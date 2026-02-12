
from financial_TSF.steps.data_loading_step import DataLoadingStep
from financial_TSF.steps.evaluation_step import EvaluationStep
from financial_TSF.steps.feature_step import FeatureStep
from financial_TSF.steps.model_changes_step import ModelChangesStep
from financial_TSF.steps.normalizing_step import ScalingStep
from financial_TSF.steps.saving_step import SavingStep
from financial_TSF.steps.splitting_step import SplittingStep
from financial_TSF.steps.training_step import TrainingStep


STEP_REGISTRY = {
    "FeatureStep": FeatureStep,
    "DataLoadingStep": DataLoadingStep,
    "NormalizingStep": ScalingStep,
    "ModelChangesStep": ModelChangesStep,
    "TrainingStep": TrainingStep,
    "EvaluationStep": EvaluationStep,
    "SplittingStep": SplittingStep,
    "SavingStep": SavingStep
}