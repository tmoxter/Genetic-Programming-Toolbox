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
    
    def variation(self, population : torch.Tensor, step_size : float = 0.00,
                local_search_only : bool = False,
                *args, **kwargs):
        """Variation operator for the evolution of the population.

        Parameters
        ----------
        population : torch.Tensor
            The population of individuals.
        """
        if local_search_only:
            x = deepcopy(population).type(torch.float)
            hidden_repr = self.encoder(x)
            new_hidden_repr = torch.zeros_like(hidden_repr)
            #old = torch.argmax(self.decoder(x, hidden_repr.unsqueeze(0), False), dim=2)
            unchanged = torch.ones(hidden_repr.shape[1], dtype=bool)
            steps = 0
            while unchanged.any(): 
                angle = torch.randn_like(hidden_repr)
                angle = torch.divide(angle, torch.norm(angle, dim=1, keepdim=True))
                new_hidden_repr[:, unchanged, :] = hidden_repr[:, unchanged, :]\
                    + angle[:, unchanged, :]*step_size
                new = torch.argmax(self.decoder(x, new_hidden_repr, False), dim=2)
                unchanged = (population.argmax(dim=2) == new).all(dim=1)
                steps += 1
                if steps > 100:
                    break

            return self.framework.syntactic_embedding(new), 0
        
        else:
            x = deepcopy(population).type(torch.float)
            hidden_repr = self.encoder(x)[-1]
            new_hidden_repr = torch.zeros_like(hidden_repr)
            old = torch.argmax(self.decoder(x, hidden_repr.unsqueeze(0), False), dim=2)
            # --- fit gaussian to the population ---
            mu = torch.mean(hidden_repr, dim=0)
            sigma = torch.std(hidden_repr, dim=0)
            # --- sample new hidden representations ---
            new_hidden_repr = torch.randn_like(hidden_repr)*sigma + mu
            new = torch.argmax(self.decoder(x, new_hidden_repr.unsqueeze(0), False), dim=2)
            return self.framework.syntactic_embedding(new), 0

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.4)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size).to(x.device)

        _, (hn, _) = self.lstm(x, (h0, c0))

        return torch.tanh(hn)

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x : torch.tensor, hx : torch.tensor,
                teacher_forcing : bool = False):
        
        decoder_sequence = torch.zeros_like(x, dtype=torch.float)
        h, c = hx, torch.rand_like(hx)
        # --- start-of-sequence token <sos> as zeros ---
        token = torch.zeros(x.shape[0], x.shape[2])
        for t in range(decoder_sequence.shape[1]):
            _, (h, c) = self.lstm(token.unsqueeze(1), (h, c))
            #token = torch.softmax(self.fc(h[self.num_layers-1]), dim=1)
            token = torch.sigmoid(self.fc(h[self.num_layers-1]))
            decoder_sequence[:, t, :] = token

            if teacher_forcing:
                token = x[:, t, :]

        return decoder_sequence