import torch
import torch.nn as nn
from copy import deepcopy

class DenseAutoencoder(nn.Module):
    """Dense Autoencoder model.

    Parameters
    ----------
    framework : Framework
        The framework object.
    input_dim : int
        The input dimension of the model.
    hidden_dim : int
        The hidden dimension of the model.
    n_decoder_heads : int, optional
        The number of decoder heads. The default is 1.
    
    Attributes
    ----------
    encoder : nn.Sequential
        The encoder module.
    decoder : list[nn.Sequential]
        The list of decoder heads.
    variation : callable
        The variation operator for the evolution."""
    
    def __init__(self, framework, input_dim, hidden_dim, n_decoder_heads : int = 1):
        super(DenseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.Tanh())
        
        self.decoder = list()
        for n in range(n_decoder_heads):
            self.__dict__["_modules"][f"deco_head{n}"] = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout(0.25),
                nn.LeakyReLU()
                )
            self.decoder.append(self.__dict__["_modules"][f"deco_head{n}"])
        
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
    
    def variation(self, population : torch.tensor,
            operator : str = "local", *args, **kwargs)-> torch.tensor:
        """Perform variation in the model's latent space.

        Parameters
        ----------
        population : torch.tensor
            The population to be varied.
        operator : str, optional
            The variation operator to be used. The default is "local".

        Returns
        -------
        offspring : torch.tensor"""
        
        self.eval()
        if operator == "crossover":
            return self._crossover(population)
        elif operator == "local":
            return self._local_variation(population)
        else:
            raise NotImplementedError(
                "Operator {} not implemented".format(operator))
    
    def _crossover(self, population : torch.tensor) -> torch.tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _choose_head(self, x : torch.Tensor, x_ : torch.Tensor) -> torch.Tensor:
        """Choose the head with the highest accuracy for the original tree.

        Parameters
        ----------
        x : torch.Tensor
            Original
        x_ : torch.Tensor
            Reconstructed
        
        Returns
        -------
        torch.Tensor
            The index of the best head for each individual in the population."""
        
        recon = torch.argmax(x_, dim=3)
        target = torch.argmax(x, dim=2)
        acc = (recon == target).type(torch.float).mean(dim=2).T
        best_head = torch.argmax(acc, dim=1)
        return best_head

    @torch.no_grad()   
    def _local_variation(self, population : torch.tensor, *args):

        x = deepcopy(population)
        reshape = x.shape
        old_recons = self.forward(population)
        idx_head = self._choose_head(population, old_recons)
        old_recons.transpose_(0, 1)
        old_reconstruction = old_recons[torch.arange(idx_head.size(0)), idx_head]
        old_seq = torch.argmax(old_reconstruction.reshape(reshape), dim=2)

        x = x.view(x.size(0), -1).type(torch.float)
        hidden_repr = self.encoder(x)
    
        new_hidden_repr = torch.zeros_like(hidden_repr)
        unchanged = torch.ones(hidden_repr.shape[0], dtype=bool)
        step_size, increase = 0, 0.0025

        while unchanged.any():
            # More efficient to dynamically adjust step size during evolution (self-adaptively) rather
            # than incrementally each time
            angle = torch.randn_like(hidden_repr)
            norm = torch.sum(angle**2, dim=1)**.5
            angle = torch.divide(angle.T, norm)
            #hidden_repr[unchanged] + angle.T[unchanged]*step_size
            new_hidden_repr[unchanged] = angle.T[unchanged]
            step_size += increase
            new_recon = torch.cat([
                        head(new_hidden_repr).unsqueeze(0) for head in self.decoder
                        ])
            new_recon = new_recon.reshape(new_recon.shape[0], *reshape)
            new_recon.transpose_(0, 1)
            new_reconstruction = new_recon[torch.arange(idx_head.size(0)), idx_head]
            new_seq = torch.argmax(new_reconstruction.reshape(reshape), dim=2)

            unchanged = (old_seq == new_seq).all(dim=1)
            if step_size > 25:
                print("Step size too large, breaking")
                break
        
        # x = x.reshape(reshape)
        # mask = [x == old_reconstruction]
        # x_new = x[mask] = new_reconstruction[mask]
        #x_new = x.reshape(reshape) + (new_reconstruction - old_reconstruction)
        
        return self.framework.syntactic_embedding(new_reconstruction), 0