import torch
import torch.nn as nn
from copy import deepcopy

class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU())
            
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LeakyReLU())
        
        self.input_dim = input_dim

    def forward(self, x : torch.tensor):
        
        reshape = x.shape
        x = x.view(x.size(0), -1)
        x = x.type(torch.float)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(reshape)
        return x
    
    def vary(self, x : torch.tensor):

        x = deepcopy(x)
        reshape = x.shape
        x = x.view(x.size(0), -1)
        hidden_repr = self.encoder(x)
        angle = torch.randn_like(hidden_repr)
        norm = torch.sum(angle**2, dim=1)**.5
        angle = torch.divide(angle.T, norm)
        new_hidden_repr = torch.zeros_like(hidden_repr)
        unchanged = torch.ones(hidden_repr.shape[0], dtype=bool)
        step_size, increase = 0, 0.025
        while unchanged.any(): 
            new_hidden_repr[unchanged] = hidden_repr[unchanged] + angle.T[unchanged]*step_size
            step_size += increase
            old = torch.argmax(self.decoder(hidden_repr).reshape(reshape), dim=2)
            new = torch.argmax(self.decoder(new_hidden_repr).reshape(reshape), dim=2)
            unchanged = (old == new).all(dim=1)
            if step_size > 20:
                break

        new_reconstruction = self.decoder(new_hidden_repr).reshape(reshape)
        old_reconstruction = self.decoder(hidden_repr).reshape(reshape)
        x_new = x.reshape(reshape) + (new_reconstruction - old_reconstruction)
        x_new = (x_new == torch.max(x_new, dim=2, keepdim=True).values).float()
        
        return x_new