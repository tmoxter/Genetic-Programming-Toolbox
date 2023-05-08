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

class SemanticData(Dataset):
    def __init__(self, population, semantics, fitnesses):
        self.population = population
        self.semantics = semantics
        self.fitnesses = fitnesses
    
    def __len__(self):
        return len(self.population)
    
    def __getitem__(self, index):
        p = self.population[index].to(torch.device("cpu"))
        s = self.semantics[index].to(torch.device("cpu"))
        f = self.fitnesses[index].to(torch.device("cpu"))
        
        return p, s, f