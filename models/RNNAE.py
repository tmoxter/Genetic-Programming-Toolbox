import torch
import torch.nn as nn
from copy import deepcopy

class RecurrentAutoEncoder(nn.Module):
    """Recurrent Autoencoder model based on LSTM modules.

    Parameters
    ----------
    framework : Framework
        The framework object.
    input_size : int
        The input dimension of the model.
    hidden_size : int
        The hidden dimension of the model.
    num_layers : int
        The number of layers of the model.
    seq_len : int
        The sequence length of the model.

    Attributes
    ----------
    encoder : nn.Module
        The encoder module.
    decoder : nn.Module
        Decoder module.
    variation : callable
        The variation operator for the evolution.
    """

    
    def __init__(self, framework, input_size, hidden_size, num_layers, seq_len):
                 
        super(RecurrentAutoEncoder, self).__init__()
        self.encoder = RNNEncoder(input_size, hidden_size, num_layers)
        self.decoder = RNNDecoder(input_size, hidden_size, num_layers)
        self.seq_len, self.num_layers = seq_len, num_layers
        self.framework = framework

    def forward(self, x, teacher_forcing = True):
        
        encoded = self.encoder(x.type(torch.float))
        decoded = self.decoder(x.type(torch.float),
                               encoded, teacher_forcing)

        return decoded
    
    def variation(self, population : torch.Tensor, *args, **kwargs):
        """Variation operator for the evolution of the population.

        Parameters
        ----------
        population : torch.Tensor
            The population of individuals.
        """
        x = deepcopy(population).type(torch.float)
        hidden_repr = self.encoder(x)[-1]
        angle = torch.randn_like(hidden_repr)
        norm = torch.sum(angle**2, dim=1)**.5
        angle = torch.divide(angle.T, norm)
        new_hidden_repr = torch.zeros_like(hidden_repr)
        unchanged = torch.ones(hidden_repr.shape[0], dtype=bool)
        step_size, increase = 0, 0.025
        while unchanged.any(): 
            new_hidden_repr[unchanged] = hidden_repr[unchanged]\
                  + angle.T[unchanged]*step_size
            step_size += increase
            old = torch.argmax(self.decoder(x, hidden_repr.unsqueeze(0), False), dim=2)
            new = torch.argmax(self.decoder(x, new_hidden_repr.unsqueeze(0), False), dim=2)
            unchanged = (old == new).all(dim=1)
            if step_size > 20:
                break

        new_reconstruction = self.decoder(x, new_hidden_repr.unsqueeze(0), False)

        return self.framework.syntactic_embedding(new_reconstruction), 0

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size).to(x.device)

        _, (hn, _) = self.lstm(x, (h0, c0))

        return hn

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x : torch.tensor, hx : torch.tensor,
                teacher_forcing : bool = False):
        
        decoder_sequence = torch.zeros_like(x, dtype=torch.float)
        h, c = hx, torch.rand_like(hx)
        # --- start-of-sequence token <sos> as zeros ---
        token = torch.zeros(x.shape[0], x.shape[2])
        for t in range(decoder_sequence.shape[1]):
            _, (h, c) = self.lstm(token.unsqueeze(1), (h, c))
            token = self.fc(h[self.num_layers-1])
            decoder_sequence[:, t, :] = token

            if teacher_forcing:
                token = x[:, t, :]

        return decoder_sequence