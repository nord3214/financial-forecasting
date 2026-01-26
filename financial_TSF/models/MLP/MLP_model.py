import copy
import random
import torch
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.batching_step import create_dataloader
from torch import nn

# inspired from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md

MODEL_CONFIG = {
    "hidden_layers": [64, 32], 
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 64
}

class MLPModel(BaseModel):
    def __init__(self, config: AppConfig, model_config: dict = MODEL_CONFIG):
        self.config = config
        self.output_dim = len(config.settings.forecast_horizon)
        self.model: nn.Module = nn.Module()
        self.hidden_layers = model_config.get("hidden_layers", [64, 32])
        self.dropout = model_config.get("dropout", 0.2)
        self.lr = model_config.get("lr", 0.001)
        self.batch_size = model_config.get("batch_size", 64)
        self.patience = 5 # for early stopping
        self.is_tune_hyperparameters = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for training")


    def fit(self, step_result: StepResult):

        train_dataloader  = create_dataloader(step_result.X_train, step_result.y_train, batch_size=self.batch_size, shuffle=True)
        # create separate val dataset
        val_loader  = create_dataloader(step_result.X_val, step_result.y_val, batch_size=self.batch_size, shuffle=False)

        # initialize model
        input_dim = step_result.X_train.shape[1]
        self.model = DynamicMLP(
            input_dim=input_dim, 
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers, 
            dropout_rate=self.dropout)
        self.model.to(self.device) # move model to device (gpu)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # loss_fn = nn.MSELoss() # think about HuberLoss/ l1 for robustness to outliers
        loss_fn = nn.HuberLoss()
        
        # train with early stopping
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        no_improve_epochs = 0
        
        
        for epoch in range(50):  # number of epochs
            print(f'Starting epoch {epoch+1}')
            
            self.model.train()
            current_loss = 0.0

            # get batched data from dataloader
            for i, data in enumerate(train_dataloader):
                inputs, targets = data
                # do float conversion and move to device!
                inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
                # adjust for multi hrizon
                targets = targets.view(targets.shape[0], self.output_dim)

                optimizer.zero_grad()

                #batched outputs
                outputs = self.model(inputs)

                # loss calculation accross each horizon then mean
                loss = loss_fn(outputs, targets)

                loss.backward()

                optimizer.step()

                # print statistics
                # accumulate loss over batches
                current_loss += loss.item()
                if (i+1) % 5000 == 0:
                    print(f'Batch {i+1}, Loss: {current_loss / 5000}')
                    current_loss = 0.0
                    
            # Validation Step
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
                    targets = targets.view(targets.shape[0], self.output_dim)
                    
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Check Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= self.patience:
                print(f"Early Stopping at Epoch {epoch+1}")
                break

        print('Training process has finished.')
        
        self.model.load_state_dict(best_model_wts)

        return best_val_loss
    
    def predict(self, step_result) -> tuple:
        
        
        self.model.eval()
        
        X_train_tensor = torch.tensor(step_result.X_train.to_numpy().astype('float32')).to(self.device)
        X_test_tensor = torch.tensor(step_result.X_test.to_numpy().astype('float32')).to(self.device)
        
        # predict without gradient calculation
        with torch.no_grad():
            # forecast on train set
            y_train_pred_tensor = self.model(X_train_tensor)
            
            # forecast on test set
            y_test_pred_tensor = self.model(X_test_tensor)
            
        y_train_pred = y_train_pred_tensor.cpu().numpy()
        y_test_pred = y_test_pred_tensor.cpu().numpy()
        
        y_train = step_result.y_train
        y_test = step_result.y_test
        
        print(f"Prediction completed. Shape of test predictions: {y_test_pred.shape}")
    
        return y_test, y_test_pred, y_train, y_train_pred
    
    def tune_hyperparameters(self, step_result: StepResult) -> dict:
        print("Starting hyperparameter tuning...")
    
        best_config = run_hyperparameter_tuning(self.config, step_result)
        
        return best_config


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int], dropout_rate: float):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU()) # maybe gelu?
            layers.append(nn.Dropout(dropout_rate)) # add dropout for better generalization
            in_dim = h_dim
            
        # Output Layer 
        # :TODO adjust for multi horizon
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


    
    
def run_hyperparameter_tuning(pipeline_config: AppConfig, step_result):
    # define search space
    param_grid = {
        "hidden_layers": [[32], [64, 32], [128, 64, 32]], 
        "dropout": [0.1, 0.2, 0.5],
        "lr": [0.001, 0.0005, 0.0001],
        "batch_size": [32, 64, 128]
    }
    
    best_loss = float('inf')
    best_config = {}
    
    # loop 10 time over random configurations
    for i in range(15):
        model_config = {
            "hidden_layers": random.choice(param_grid["hidden_layers"]),
            "dropout": random.choice(param_grid["dropout"]),
            "lr": random.choice(param_grid["lr"]),
            "batch_size": random.choice(param_grid["batch_size"]),
            "epochs": 50,
            "patience": 5
        }
        
        print(f"Testing Config {i}: {model_config}")
        
        # modell training using validation set
        model = MLPModel(pipeline_config, model_config=model_config)
        final_val_loss = model.fit(step_result) 
        print(f"Validation Loss for Config {i}: {final_val_loss}")
        
        # if current loss ist better(lower), save config and loss
        if final_val_loss < best_loss:
           best_loss = final_val_loss
           best_config = model_config
            
    print("Winner:", best_config, "with best loss:", best_loss)
    return best_config