from torch.utils.data import Dataset, DataLoader



# create custome dataset 
class TickerDataset(Dataset):

    def __init__(self, X_test, y_test):
        self.X_test = X_test.to_numpy()
        self.y_test = y_test.to_numpy()

        print("Received data for Dataloader with", len(self), "samples.")
    
    def __len__(self):
        return len(self.X_test)
    
    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]
    

def create_dataloader(X_test, y_test, batch_size):
    dataset = TickerDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


# create dataloader

# do batching 