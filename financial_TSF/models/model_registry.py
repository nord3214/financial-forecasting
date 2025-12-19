from financial_TSF.models.linear_regression.linear_regression_model import LinearRegressionModel
from financial_TSF.models.rf.random_forest_regressor import RandomForestRegressor


MODEL_REGISTRY = {
    "RF": RandomForestRegressor,
    "LinearRegression": LinearRegressionModel
}