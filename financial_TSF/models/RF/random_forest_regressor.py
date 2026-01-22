from typing import Any, Optional
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel
from scipy.stats import randint



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
        
        self.X_train: Optional[pd.DataFrame]
        self.X_test: Optional[pd.DataFrame] 
        self.y_train: Optional[pd.Series] 
        self.y_test: Optional[pd.Series] 
        
    def fit(self, step_result: StepResult):
        
        self.X_train = step_result.X_train
        self.X_test = step_result.X_test
        self.y_train = step_result.y_train
        self.y_test = step_result.y_test
        
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data splits are not prepared. Call prepare_data() or Datasplit Pipeline step first.")
        
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(
            n_jobs= -2,
            random_state=1337
        )
        tune_hyperparameters = True
        
        if tune_hyperparameters:
            print("start  Hyperparameter-Tuning (RandomizedSearch)...")
                        
            # different parameters to tune
            param_dist = {
                "n_estimators": randint(100, 500),      
                "max_depth": [10, 20, 30, None],       
                "min_samples_leaf": randint(20, 100),  # higher value helps against noise
                "min_samples_split": randint(2, 20),
                "max_features": ["sqrt", "log2", None]
            }

            
            # tuning with time series split cross-validation so no shuffle happens
            tscv = TimeSeriesSplit(n_splits=3)

            # randomized Search
            search = RandomizedSearchCV(
                estimator=rfr,
                param_distributions=param_dist,
                n_iter=15,              # try n_iter random combinations
                scoring="neg_mean_squared_error",
                cv=tscv,
                n_jobs=-2,
                verbose=1,
                random_state=1337
            )
            
            # for now only use part of train data 
            # print("Head of X_train: ", self.X_train.iloc[-20000:].head(25))
            # search.fit(self.X_train.iloc[-1000:], self.y_train.iloc[-1000:])
            
            search.fit(self.X_train, self.y_train)

            print(f"Best parameters found: {search.best_params_}")
            with open("rfr_best_params.txt", "w") as f:
                f.write(str(search.best_params_))
                
            self.estimator = search.best_estimator_
            
            # # Save the best params for later use
            # self.best_params = search.best_params_
            
        else:
            # use hardcoded hyperparameters
            final_params = HYPERPARAMS
            
            self.estimator = rfr.set_params(**final_params)
            
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
        