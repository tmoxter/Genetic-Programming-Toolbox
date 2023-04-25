import torch
import torch.nn as nn
from copy import deepcopy

class RecurrentAutoEncoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
                 
        super(RecurrentAutoEncoder, self).__init__()
        self.encoder = RNNEncoder(input_size, hidden_size, num_layers)
        self.decoder = RNNDecoder(input_size, hidden_size, num_layers)
        self.seq_len = seq_len
        self.num_layers = num_layers

    def forward(self, x, teacher_forcing = True):
        
        encoded = self.encoder(x.type(torch.float))
        decoded = self.decoder(x.type(torch.float),
                               encoded, teacher_forcing)

        return decoded

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