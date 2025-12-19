from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel




class LinearRegressionModel(BaseModel):
    def __init__(self, config: AppConfig):
        self.config = config    
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def fit(self, step_result: StepResult):
        
        self.X_train = step_result.X_train
        self.X_test = step_result.X_test
        self.y_train = step_result.y_train
        self.y_test = step_result.y_test
        
        # X_scaled = self.scaler.fit_transform(step_result.X_train)
        X_scaled = step_result.X_train  # Linear Regression may not need scaling
        self.model.fit(X_scaled, step_result.y_train)

    def predict(self, step_result: StepResult):
                
        y_test_pred = self.model.predict(self.X_test)
        y_train_pred = self.model.predict(self.X_train)

        
        return self.y_test, y_test_pred, self.y_train, y_train_pred
