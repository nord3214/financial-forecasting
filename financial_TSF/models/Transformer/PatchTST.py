from typing import Any
import pandas as pd
from neuralforecast.models import PatchTST
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel



class PatchTSTWrapper(BaseModel):
    def __init__(self, config):

        self.model = PatchTST(
            h=12,                   # Forecast-horizon in steps
            input_size=104,         # Lookback Window
            patch_len=16,           # length of patches
            stride=8,               # Stride
            loss=MAE(),             # Loss function for training
            valid_loss=MAE(),       # Loss function for validation
            max_steps=1000,         # Maximum training steps
            early_stop_patience_steps=10, # Stops if val-loss doesn't improve for x checks
            val_check_steps=50,     # Check validation every 50 steps
            scaler_type=None,# Data already scaled 
            revin=True,             # (Reversible Instance Norm) if non starionary checks  are not sufficent
            batch_size=32
        )

        self.nf = NeuralForecast(
            models=[self.model],
            freq='5min'
        )
    
    def _prepare_dataframe(self, X, y):
        """
        prepare df for the specific columns
        Columns: 'unique_id', 'ds', 'y' and possibly features
        """
        df = pd.concat([X, y], axis=1)
        
        # If your data does not have 'unique_id'
        if 'unique_id' not in df.columns:
            raise ValueError("For PatchTST, 'unique_id' column is required. Please ensure your data includes this column for each time series (e.g., ticker).")
        

        return df

    def fit(self, step_result: StepResult) -> Any:
        # put into correct column format
        train_data = self._prepare_dataframe(step_result.X_train, step_result.y_train)
        
        # Fit with Validation Size
        # NeuralForecast uses the last 'val_size' time points for validation
        self.nf.fit(df=train_data, val_size=14) 
        
        print(f"Fitting PatchTST completed. Training shape: {train_data.shape}")
    
    def predict(self, step_result: StepResult) -> tuple:

        forecasts = self.nf.predict()
        
        
        return forecasts, step_result.y_val # Example return
    
