import torch
from torch.utils.data import Dataset

class PopulationData(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_data = self.data[index].to(torch.device("cpu"))
        
        return input_data