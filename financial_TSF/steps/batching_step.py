import torch
from torch.utils.data import Dataset, DataLoader

# create custome dataset 
class TickerDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X.to_numpy()).float()
        self.y = torch.tensor(y.to_numpy()).float()

        print("Received data for Dataloader with", len(self), "samples.")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def create_dataloader(X, y, batch_size, shuffle=False):
    dataset = TickerDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader