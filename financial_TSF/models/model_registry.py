from financial_TSF.models.MLP.MLP_model import MLPModel
from financial_TSF.models.constant.constant_model import ConstantModel
from financial_TSF.models.Transformer.PatchTST import PatchTSTWrapper
from financial_TSF.models.linear_regression.linear_regression_model import LinearRegressionModel
from financial_TSF.models.rf.random_forest_regressor import RandomForestModel
from financial_TSF.models.naive.naive_model import NaiveModel


MODEL_REGISTRY = {
    "RF": RandomForestModel,
    "LinearRegression": LinearRegressionModel,
    "MLP": MLPModel,
    "Constant": ConstantModel,
    "PatchTST": PatchTSTWrapper,
    "Naive": NaiveModel,
}