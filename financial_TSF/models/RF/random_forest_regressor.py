from typing import Any
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.model_selection import train_test_split



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

class RandomForestRegressor:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.rfr = None  # Placeholder for the actual RandomForestRegressor model
        
        
        
    def model(self, data: list[pd.DataFrame]) -> bool:
        """Model Closing price from Indicators."""
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
        y_pred_as_lag = df["close"].shift(SETTINGS["forecast_horizon"])
        
          
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(**HYPERPARAMS)
        
        if SETTINGS["use_RFECV"]:
            tscv = TimeSeriesSplit(RFECV_SETTINGS["cv_splits"])
            
            selector = RFECV(
                estimator=rfr,
                cv=tscv,
                step=RFECV_SETTINGS["step"],
                scoring=RFECV_SETTINGS["scoring"],
            )
        
            selector.fit(X_train, y_train)
        else:
            rfr.fit(X_train, y_train)
            selector = rfr
        
        y_pred = selector.predict(X_test)
        
        # :TODO Put evaluation metrics in a seperate step
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        print("Mean Absolute Error (MAE): ", mean_absolute_error(y_test, y_pred))
        print("Mean Sqaured Error (MAE): ", mean_squared_error(y_test, y_pred))
        print("R2 Score: ", r2_score(y_test, y_pred))
        
        y_pred_as_lag = y_pred_as_lag[-len(y_pred):]
        
        return True
    
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
    
        
        
        
RandomForestRegressor("test").model([pd.DataFrame()])