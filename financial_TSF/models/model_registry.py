from financial_TSF.models.MLP.MLP_model import MLPModel
from financial_TSF.models.Transformer.PatchTST import PatchTSTWrapper
from financial_TSF.models.linear_regression.linear_regression_model import LinearRegressionModel
from financial_TSF.models.rf.random_forest_regressor import RandomForestModel


MODEL_REGISTRY = {
    "RF": RandomForestModel,
    "LinearRegression": LinearRegressionModel,
    "MLP": MLPModel,
    "PatchTST": PatchTSTWrapper
}