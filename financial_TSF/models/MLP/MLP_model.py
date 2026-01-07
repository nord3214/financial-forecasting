

import torch
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.models.base_model import BaseModel
from financial_TSF.steps.batching_step import create_dataloader
from torch import nn

# inspired from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md

class MLPModel(BaseModel):
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, step_result: StepResult):

        train_dataloader  = create_dataloader(step_result.X_train, step_result.y_train, batch_size=8)

        for epoch in range(10):  # number of epochs
            print(f'Starting epoch {epoch+1}')

            current_loss = 0.0

            # get batched data from dataloader
            for i, data in enumerate(train_dataloader):
                inputs, targets = data
                # do float conversion
                inputs, targets = inputs.float(), targets.float()
                # adjust for multi hrizon
                targets = targets.reshape((targets.shape[0], 1))

                self.optimizer.zero_grad()

                #batched outputs
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, targets)

                loss.backward()

                self.optimizer.step()

                # print statistics
                # accumulate loss over batches
                current_loss += loss.item()
                if (i+1) % 50 == 0:
                    print(f'Batch {i+1}, Loss: {current_loss / 50}')
                    current_loss = 0.0

        print('Training process has finished.')

        return 
    
    def predict(self, step_result) -> tuple:
        pass

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)
    