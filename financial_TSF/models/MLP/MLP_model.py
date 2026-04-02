import copy
from pathlib import Path
import random
import pandas as pd
import torch
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.MLP.model_params import MODEL_PARAMS
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.batching_step import create_dataloader
from torch import nn

from financial_TSF.utils.utils import get_result_path, print_target_diagnostics, print_prediction_diagnostics, plot_learning_curve, save_param_dict, save_tuning_results

# inspired from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md




class MLPModel(BaseModel):
    def __init__(self, config: AppConfig, model_config: dict = MODEL_PARAMS):
        self.config = config
        self.output_dim = len(config.settings.forecast_horizon)
        self.model: nn.Module = nn.Module()
        self.hidden_layers = model_config.get("hidden_layers", [64, 32])
        self.dropout = model_config.get("dropout", 0.2)
        self.lr = model_config.get("lr", 0.001)
        self.weight_decay = model_config.get("weight_decay", 1e-4)
        self.batch_size = model_config.get("batch_size", 64)
        self.patience = model_config.get("patience", 30) # for early stopping
        self.iteration = 0
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for training")


    def fit(self, step_result: StepResult, refit: bool = True):
        
        
        save_param_dict(self.model_config, self.config, step_result, iteration=self.iteration)
        
        train_losses = []
        val_losses = []

        train_dataloader  = create_dataloader(step_result.X_train, step_result.y_train, batch_size=self.batch_size, shuffle=True)
        # create separate val dataset
        val_loader  = create_dataloader(step_result.X_val, step_result.y_val, batch_size=self.batch_size, shuffle=False)

        # print_target_diagnostics(step_result.y_train)

        # initialize model
        input_dim = step_result.X_train.shape[1]
        self.model = DynamicMLP(
            input_dim=input_dim, 
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers, 
            dropout_rate=self.dropout)
        self.model.to(self.device) # move model to device (gpu)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # loss_fn = nn.MSELoss()
        loss_fn = nn.L1Loss() # MAE - more robust to outliers in financial returns
        # loss_fn = nn.HuberLoss()
        
        # train with early stopping
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        no_improve_epochs = 0
        best_epoch = 1
        
        
        for epoch in range(400):  # number of epochs
            print(f'Starting epoch {epoch+1}')
            
            self.model.train()
            train_loss = 0.0

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

                # gradient diagnostics
                if i == 0 and (epoch + 1) % 5 == 0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            print(f"[Grad] {name:40s}  norm={grad_norm:.2e}")

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            print(f'Epoch {epoch+1} Train Loss: {train_loss:.6f}')
                    
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
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1} completed. Validation Loss: {val_loss:.6f}')
            
            # Check Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
                best_epoch = epoch + 1
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= self.patience:
                print(f"Early Stopping at Epoch {epoch+1}")
                break
            
            print_prediction_diagnostics(self.model, val_loader, train_dataloader, self.device, epoch + 1)

        print('Training process has finished.')
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        self.model.load_state_dict(best_model_wts)
        refit = False
        if refit:
            # final refit on combined train+val
            print(f"[MLP] Refitting on train+val combined for {best_epoch} epochs...")
            X_combined = pd.concat([step_result.X_train, step_result.X_val])
            y_combined = pd.concat([step_result.y_train, step_result.y_val])
            combined_loader = create_dataloader(
                X_combined, y_combined, batch_size=self.batch_size, shuffle=True
            )

            self.model = DynamicMLP(
                input_dim=input_dim,
                output_dim=self.output_dim,
                hidden_layers=self.hidden_layers,
                dropout_rate=self.dropout,
            )
            self.model.to(self.device)
            optimizer_refit = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            for ep in range(best_epoch):
                self.model.train()
                for data in combined_loader:
                    inputs, targets = data
                    inputs  = inputs.to(self.device).float()
                    targets = targets.to(self.device).float()
                    targets = targets.view(targets.shape[0], self.output_dim)
                    optimizer_refit.zero_grad()
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer_refit.step()
                print(f"[MLP Refit] Epoch {ep + 1}/{best_epoch}")

            self.model.eval()
            print("[MLP] Refit complete.")
        save_path = get_result_path(self.config, step_result)
            
        plot_learning_curve(train_losses, val_losses, model_name="MLP", save_path=save_path, iteration=self.iteration)
        
        self.save_model(step_result)

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
        """
        random search over hyperparameters using X_val / y_val
        The best config is written back into
        self so the subsequent fit() call uses it.
        """
        print("[MLP] Starting hyperparameter tuning with validation set...")

        param_grid = {
            "hidden_layers": [[512, 256, 128, 64], [256, 256, 128, 64], [256, 256, 128], [256, 128, 64],
                              [128, 64, 32], [512, 256, 128], [128, 128]],
            "dropout": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
            "lr": [1e-3, 5e-4, 1e-4],
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            "batch_size": [64, 128, 256, 512,],
        }

        n_iter = 10
        best_loss = float("inf")
        best_config: dict = {}
        all_trials: list[dict] = []

        for i in range(n_iter):
            model_config = {
                "hidden_layers": random.choice(param_grid["hidden_layers"]),
                "dropout":       random.choice(param_grid["dropout"]),
                "lr":            random.choice(param_grid["lr"]),
                "weight_decay":  random.choice(param_grid["weight_decay"]),
                "batch_size":    random.choice(param_grid["batch_size"]),
                "patience":      40,
            }

            print(f"\n  [{i+1:>2}/{n_iter}] Config: {model_config}")

            candidate = MLPModel(self.config, model_config=model_config)
            candidate.iteration = i
            val_loss = candidate.fit(step_result, refit=False)
            print(f"  [{i+1:>2}/{n_iter}] val_loss={val_loss:.6f}")
            all_trials.append({"params": model_config, "val_loss": val_loss})

            if val_loss < best_loss:
                best_loss = val_loss
                best_config = model_config

        print(f"\n[MLP] Best val_loss={best_loss:.6f}")
        print(f"[MLP] Best config: {best_config}")

        # update self so the next fit() uses the best config
        self.hidden_layers = best_config["hidden_layers"]
        self.dropout       = best_config["dropout"]
        self.lr            = best_config["lr"]
        self.weight_decay  = best_config["weight_decay"]
        self.batch_size    = best_config["batch_size"]

        save_tuning_results(all_trials, best_config, "MLP", self.config, step_result)

        return {"best_config": best_config, "val_loss": best_loss}
    
    def save_model(self, step_result: StepResult) -> Path:
        result_dir = get_result_path(self.config, step_result) / "saved_model"
        result_dir.mkdir(parents=True, exist_ok=True)
        # Save weights + the one missing architecture param
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.model.network[0].in_features,  # extract from first Linear layer
            # "input_dim": step_result.X_train.shape[1]
        }, result_dir / "model.pt")
        print(f"[MLP] Model saved to {result_dir}")
        return result_dir

    def load_model(self, model_path: Path) -> None:
        checkpoint = torch.load(model_path, weights_only=True)
        self.model = DynamicMLP(
            input_dim=checkpoint["input_dim"],
            output_dim=self.output_dim,       # already in self from __init__
            hidden_layers=self.hidden_layers,  # already in self from __init__
            dropout_rate=self.dropout,         # already in self from __init__
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[MLP] Model loaded from {model_path}")


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int], dropout_rate: float):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())  # GELU avoids dying ReLU — smooth gradient for near-zero activations
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        # Output Layer 
        # :TODO adjust for multi horizon
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


    
    
# def run_hyperparameter_tuning(pipeline_config: AppConfig, step_result):
#     # define search space
#     param_grid = {
#         "hidden_layers": [[256, 256, 128, 64], [256, 256, 128], [256, 128, 64]],
#         "dropout": [0.1, 0.2, 0.5],
#         "lr": [0.001, 0.0005, 0.0001],
#         "batch_size": [512, 1024]
#     }
    
#     best_loss = float('inf')
#     best_config = {}
    
#     # loop 10 time over random configurations
#     for i in range(15):
#         model_config = {
#             "hidden_layers": random.choice(param_grid["hidden_layers"]),
#             "dropout": random.choice(param_grid["dropout"]),
#             "lr": random.choice(param_grid["lr"]),
#             "batch_size": random.choice(param_grid["batch_size"]),
#             "epochs": 50,
#             "patience": 20
#         }
        
#         print(f"Testing Config {i}: {model_config}")
        
#         # modell training using validation set
#         model = MLPModel(pipeline_config, model_config=model_config)
#         final_val_loss = model.fit(step_result) 
#         print(f"Validation Loss for Config {i}: {final_val_loss}")
        
#         # if current loss ist better(lower), save config and loss
#         if final_val_loss < best_loss:
#            best_loss = final_val_loss
#            best_config = model_config
            
#     print("Winner:", best_config, "with best loss:", best_loss)
#     return best_config