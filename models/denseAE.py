import torch
import torch.nn as nn
from copy import deepcopy

class DenseAutoencoder(nn.Module):
    def __init__(self, framework, input_dim, hidden_dim, n_decoder_heads : int = 1):
        super(DenseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh())
        
        for n in range(n_decoder_heads):
            self.__dict__["_modules"][f"deco_head{n}"] = nn.Sequential(
                nn.Linear(hidden_dim, input_dim), nn.LeakyReLU()
                )

        self.decoder = [self.__dict__["_modules"][f"deco_head{n}"]
                          for n in range(n_decoder_heads)]
        
        self.input_dim = input_dim
        self.framework = framework
        self.n_decoder_heads = n_decoder_heads

    def forward(self, x : torch.tensor):
        
        reshape = x.shape
        x = x.view(x.size(0), -1).type(torch.float)
        x_ = self.encoder(x)
        
        y = torch.cat([
            head(x_).unsqueeze(0) for head in self.decoder
            ])

        y = y.reshape(y.shape[0], *reshape)
        return y
    
    def variation(self, population : torch.tensor, *args):

        x = deepcopy(population)
        reshape = x.shape

        old_recons = self.forward(population)
        population = population.expand(old_recons.shape[0], -1, -1, -1)
        population = population.type(torch.float)
        p = (1, 3, 0, 2) # -> reduction='none' applies cross_entropy along dim=1
        sample_loss = nn.functional.cross_entropy(old_recons.permute(*p),
                                population.permute(*p), reduction='none')
        sample_loss = sample_loss.mean(dim = -1) # -> loss now of shape (n_samples, n_heads)
        idx_head = torch.argmin(sample_loss, dim=1)
        old_recons = old_recons.permute(1,0,2,3)
        old_reconstruction = old_recons[torch.arange(idx_head.size(0)), idx_head]
        old_seq = torch.argmax(old_reconstruction.reshape(reshape), dim=2)

        x = x.view(x.size(0), -1).type(torch.float)
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

            new_recon = torch.cat([
                        head(new_hidden_repr).unsqueeze(0) for head in self.decoder
                        ])
            new_recon = new_recon.reshape(new_recon.shape[0], *reshape)
            new_recon = new_recon.permute(1, 0, 2, 3)
            new_reconstruction = new_recon[torch.arange(idx_head.size(0)), idx_head]
            new_seq = torch.argmax(new_reconstruction.reshape(reshape), dim=2)

            unchanged = (old_seq == new_seq).all(dim=1)
            if step_size > 20:
                break

        x_new = x.reshape(reshape) + (new_reconstruction - old_reconstruction)
        
        return self.framework.syntactic_embedding(x_new)