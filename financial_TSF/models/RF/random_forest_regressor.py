from typing import Any, Optional, cast
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.model_selection import train_test_split

from financial_TSF.models.base_model import BaseModel



HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "random_state": 17,
    "n_jobs": -1,
    "max_features": "sqrt",
    "min_samples_leaf": 40,
    "max_depth": 15,
}

SETTINGS: dict[str, Any] = {
    "forecast_horizon": 12,
    "use_RFECV": True,
}

RFECV_SETTINGS: dict[str, Any] = {
    "step": 1,
    "cv_splits": 5,
    "scoring": "neg_mean_squared_error",
}

class RandomForestRegressor(BaseModel):
    def __init__(self, config: Any) -> None:
        self.config = config
        self.estimator: Optional[Any] = None  # placeholder fitted estimator model
        
        # Train/test splits populated in prepare_data()
        self.X_train: Optional[pd.DataFrame]
        self.X_test: Optional[pd.DataFrame] 
        self.y_train: Optional[pd.Series] 
        self.y_test: Optional[pd.Series] 

        
        
        
    def prepare_data(self):
        
        # print("Modeling Close Price of Apple Test dataset with Random Forest Regressor")
        
        df = pd.read_csv("data\processed\AAPL\AAPL_intraday_5min_processed.csv")
        
        df["target"] = df["close"].shift(-SETTINGS["forecast_horizon"])
        # print("Shape before dropping NA: ", df.shape)
        df = df.dropna()
        # print("Shape after dropping NA: ", df.shape)
        # print(df.head(5))
        df = df.drop("ticker", axis=1)
        # print("Shape after dropping ticker: ", df.shape)

        X = df.drop("target", axis=1)    
        y = df["target"]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        
        
        
    def fit(self, data):
        
        self.prepare_data()
        
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(**HYPERPARAMS)
        
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data splits are not prepared. Call prepare_data() or Datasplit Pipeline step first.")
        
        if SETTINGS["use_RFECV"]:
            tscv = TimeSeriesSplit(RFECV_SETTINGS["cv_splits"])
            
            selector = RFECV(
                estimator=rfr,
                cv=tscv,
                step=RFECV_SETTINGS["step"],
                scoring=RFECV_SETTINGS["scoring"],
            )
        
            self.estimator = selector.fit(self.X_train, self.y_train)
        else:
            rfr.fit(self.X_train, self.y_train)
            self.estimator = rfr
        
    def predict(self, data):
        if self.estimator is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        if self.X_test is None or self.X_train is None or self.y_test is None or self.y_train is None:
            raise RuntimeError("Data splits are not prepared. Call prepare_data()/fit() first.")

       

        print("Mean Absolute Error (MAE): ", mean_absolute_error(self.y_test, self.estimator.predict(self.X_test)))
        print("Mean Sqaured Error (MSE): ", mean_squared_error(self.y_test, self.estimator.predict(self.X_test)))
        print("R2 Score: ", r2_score(self.y_test, self.estimator.predict(self.X_test)))

        return self.y_test, self.estimator.predict(self.X_test), self.y_train, self.estimator.predict(self.X_train)
        


        
    
    # def model_dow_jones(self):
    #     """Model Dow Jones dataset using Random Forest Regressor."""
    #     print("Modeling Close Price of Dow Jones dataset with Random Forest Regressor")
        
    #     print(sns.get_dataset_names())
        
    #     dowjones = sns.load_dataset("dowjones")
    #     print(dowjones.head())
    #     dowjones["Target"] = dowjones["Price"].shift(-1)
    #     dowjones = dowjones.drop(["Date"], axis=1)

    #     X = dowjones.drop("Price", axis=1)     
    #     y = dowjones["Price"]
          
    #     from sklearn.model_selection import train_test_split
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83, shuffle=False)
       
    #     from sklearn.ensemble import RandomForestRegressor
    #     rfr = RandomForestRegressor(n_estimators=100, random_state=83)
        
    #     rfr.fit(X_train, y_train)
        
    #     y_pred = rfr.predict(X_test)
        
    #     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
    #     print("Mean Absolute Error (MAE): ", mean_absolute_error(y_test, y_pred))
    #     print("Mean Sqaured Error (MAE): ", mean_squared_error(y_test, y_pred))
    #     print("R2 Score: ", r2_score(y_test, y_pred))
       
    #     return True
    
    # def model_life_expectancy(self):

        
    #     print(sns.get_dataset_names())
        
    #     healthexp = sns.load_dataset("healthexp")
    #     print(healthexp.head())
        
    #     healthexp = pd.get_dummies(healthexp)

    #     X = healthexp.drop("Life_Expectancy", axis=1)     
    #     y = healthexp["Life_Expectancy"]
          
    #     from sklearn.model_selection import train_test_split
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)
       
    #     from sklearn.ensemble import RandomForestRegressor
    #     rfr = RandomForestRegressor(n_estimators=100, random_state=32)
        
    #     rfr.fit(X_train, y_train)
        
    #     y_pred = rfr.predict(X_test)
        
    #     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
    #     print("Mean Absolute Error (MAE): ", mean_absolute_error(y_test, y_pred))
    #     print("Mean Sqaured Error (MAE): ", mean_squared_error(y_test, y_pred))
    #     print("R2 Score: ", r2_score(y_test, y_pred))
       
    #     return True
    
        
        
