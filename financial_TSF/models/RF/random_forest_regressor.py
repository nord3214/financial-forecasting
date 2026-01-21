from typing import Any, Optional
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.model_selection import train_test_split

from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel



HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "random_state": 17,
    "n_jobs": -2,
    "max_features": "sqrt",
    "min_samples_leaf": 40,
    "max_depth": 15,
}

SETTINGS: dict[str, Any] = {
    "forecast_horizon": 12,
    "use_RFECV": False,
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

        
        
        
    def fit(self, step_result: StepResult):
        
        # self.prepare_data()
        self.X_train = step_result.X_train
        self.X_test = step_result.X_test
        self.y_train = step_result.y_train
        self.y_test = step_result.y_test
        
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(**HYPERPARAMS)
        # rfr = MultiOutputRegressor(rfr)
        
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

            print(f"Optimal number of features: {selector.n_features_}")
            import matplotlib.pyplot as plt

            data = {
                key: value
                for key, value in selector.cv_results_.items()
                if key in ["n_features", "mean_test_score", "std_test_score"]
            }
            cv_results = pd.DataFrame(data)
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Mean test accuracy")
            plt.errorbar(
                x=cv_results["n_features"],
                y=cv_results["mean_test_score"],
                yerr=cv_results["std_test_score"],
            )
            plt.title("Recursive Feature Elimination \nwith correlated features")
            plt.show()
            
        else:
            rfr.fit(self.X_train, self.y_train)
            self.estimator = rfr
            importances = pd.Series(rfr.feature_importances_, index=self.X_train.columns)
            
            importances = importances.sort_values(ascending=False)
            print("Feature importances:\n", importances)
            
            
        
    def predict(self, step_result: StepResult) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        if self.estimator is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        if self.X_test is None or self.X_train is None or self.y_test is None or self.y_train is None:
            raise RuntimeError("Data splits are not prepared. Call prepare_data()/fit() first.")

        y_test_pred = self.estimator.predict(self.X_test)
        y_train_pred = self.estimator.predict(self.X_train)
        
        # calculate mean and variance only distribution of epistemic uncertainty
        
        # tree_preds = np.array([tree.predict(self.X_test) for tree in self.estimator.estimators_])
        # mean_pred = np.mean(tree_preds, axis=0)
        # variance_pred = np.var(tree_preds, axis=0)
        
        # feature importance 
        



        return self.y_test, y_test_pred, self.y_train, y_train_pred
        


        
    
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
    
        
        
