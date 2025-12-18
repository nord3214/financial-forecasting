from financial_TSF.models.LinearRegression.linear_regression_model import LinearRegressionModel
from financial_TSF.models.RF.random_forest_regressor import RandomForestRegressor


MODEL_REGISTRY = {
    "RF": RandomForestRegressor,
    "LinearRegression": LinearRegressionModel
}