import torch
import torch.nn as nn
from copy import deepcopy

class RecurrentVarAutoEncoder(nn.Module):
    """Recurrent Variational Autoencoder model based on LSTM modules.

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

    def __init__(self, framework, input_size, hidden_size, latent_size,
                num_layers, seq_len, dropout = 0.3, operator = 'local',
                **kwargs):
                 
        super(RecurrentVarAutoEncoder, self).__init__()

        self.operator = operator
        #latent_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = RNNEncoder(input_size = input_size,
                        hidden_size = hidden_size, num_layers = num_layers,
                        dropout = dropout)
        self.decoder = RNNDecoder(input_size = latent_size,
                        hidden_size = hidden_size, output_size = input_size,
                        num_layers = num_layers, dropout = dropout)
        
        self.fc21 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)
        self.seq_len, self.num_layers = seq_len, num_layers
        self.framework = framework
        
    def forward(self, x, teacher_forcing = True):
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        teacher_forcing : bool, optional
            Whether to use teacher forcing or not. The default is True.
        
        Returns
        -------
        decoded : torch.Tensor
            The decoded tensor.
        mu : torch.Tensor
            The mean of the latent distribution.
        logvar : torch.Tensor
            The log variance of the latent distribution.
        """

        batch_size, seq_len, _ = x.shape
        encoded, _ = self.encoder(x.type(torch.float))
        mu, logvar = self.fc21(encoded[-1]), self.fc22(encoded[-1])
        z = self.reparametize(mu, logvar)
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size)
        #z = z.repeat(self.num_layers, 1, 1)
        decoded, _ = self.decoder(z, encoded,
                                   teacher_forcing)    
        return decoded, mu, logvar
        
    def reparametize(self, mu, logvar):
        """Reparametization trick for the VAE.

        Parameters
        ----------
        mu : torch.Tensor
            The mean of the latent distribution.
        logvar : torch.Tensor
            The log variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            The reparametized latent vector.
        """
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def variation(self, population : torch.Tensor = None,
            semantics : torch.Tensor = None,
            fitnesses : torch.Tensor = None,
            operator : callable = "local",
            step_size = 1, *args, **kwargs) -> torch.Tensor:
        """Variation operator for the evolution of the population.

        Parameters
        ----------
        population : torch.Tensor
            The population of individuals.
        semantics : torch.Tensor
            The semantics of the population.
        fitnesses : torch.Tensor
            The fitnesses of the population.
        operator : callable
            The variation operator to use.
        step_size : float
            The step size of the variation operator.

        Returns
        -------
        torch.Tensor
            The offspring population.
        """
        self.eval()
        if operator == "mixed":
            ranking = fitnesses.argsort(dim=0, descending=True)
            local = ranking[:int(population.size(0)*.75)]
            offspring, offspring_semantics, n_eval = self.variation(
                population[local],semantics[local], fitnesses[local],
                operator = "local", step_size = step_size)
            cut = population.size(0) - int(population.size(0)*.75)
            crossover = ranking[:int(cut)]
            offspring_, offspring_semantics_, n_eval_ = self.variation(
                population[crossover],semantics[crossover],
                fitnesses[crossover],
                operator = "crossover", step_size = 2.5
                )
            offspring = torch.cat((offspring, offspring_), dim=0)
            offspring_semantics = torch.cat((offspring_semantics,
                                             offspring_semantics_), dim=0)
            n_eval += n_eval_
            return offspring, offspring_semantics, n_eval

        if operator == "local":

            semantic_embedding = self.framework.semantic_embedding(semantics)
            ratio = 0.5
            x = torch.clamp_max(semantic_embedding*ratio+population*(1-ratio),
                                1)

            encoded, _ = self.encoder(x.type(torch.float))
            mu, logvar = self.fc21(encoded[-1]), self.fc22(encoded[-1])
            std = torch.exp(0.5 * logvar)

            # --- sample from the distribution having latent parameters mu, var ---
            # --- equivalent to reparametrization ---
            x_ = mu + std * torch.randn_like(std) * step_size
            x_ = x_.repeat(1, population.size(1), 1)
            x_ = x_.view(population.size(0), population.size(1),
                        self.latent_size)
            decoded, _ = self.decoder(x_, encoded)
            offspring = self.framework.syntactic_embedding(decoded)
            offspring = self._protect(offspring)

            offspring_semantics = self.framework.evaluate(offspring)
        
            n_eval = self.framework.treeshape[0] \
                    * self.framework.treeshape[1]/2

            return offspring, offspring_semantics, n_eval
        
        if operator == "crossover":
            semantic_embedding = self.framework.semantic_embedding(semantics)
            ratio = 0.5
            x = torch.clamp_max(semantic_embedding*ratio+population*(1-ratio),
                                1)

            encoded, _ = self.encoder(x.type(torch.float))
            mu, logvar = self.fc21(encoded[-1]), self.fc22(encoded[-1])
            p = torch.randperm(semantics.size(0))
            mu, logvar, encoded = mu[p], logvar[p], encoded[:,p,:]
            std = torch.exp(0.5 * logvar)
            parent_pairs = torch.tensor([(i, i + mu.size(0)//2) 
                            for i in range(mu.size(0)//2)]).type(torch.long)
            idxs = torch.arange(semantics.size(0)//2).type(torch.long)
            mu_ = (mu[parent_pairs[idxs, 0]] + mu[parent_pairs[idxs, 1]])/2
            logvar_ = (logvar[parent_pairs[idxs, 0]] \
                       + logvar[parent_pairs[idxs, 1]])/2
            std_ = torch.exp(0.5 * logvar_)
            z = torch.cat((mu_ + std_ * torch.randn_like(std_)*step_size,
                        mu_ + std_ * torch.randn_like(std_)*step_size))
            z = z.repeat(1, self.seq_len, 1)
            z = z.view(mu.size(0), self.seq_len, self.latent_size)
            decoded, _ = self.decoder(z, encoded)
            offspring = self.framework.syntactic_embedding(decoded)
            offspring = self._protect(offspring)

            offspring_semantics = self.framework.evaluate(offspring)
        
            n_eval = self.framework.treeshape[0] \
                    * self.framework.treeshape[1]/2

            return offspring, offspring_semantics, n_eval

    def _protect(self, offspring : torch.Tensor):
        # --- --- syntactic protection and rejection --- ---                            
        leaves = offspring.argmax(dim=2)[:, -self.framework.leaf_info[0]:]
        leaf_primitive = self.framework.treeshape[1] \
                    - self.framework.leaf_info[1]
        refuse = (leaves < leaf_primitive).any(dim=1)
        new_leaves = self.framework.new_population(refuse.sum().item())
        offspring[refuse, -self.framework.leaf_info[0]:, :] = new_leaves[:,
                                         -self.framework.leaf_info[0]:, :]
        return offspring
    
    def latent(self, population : torch.Tensor,
               semantics : torch.Tensor)-> torch.Tensor:
        """
        Computes the latent representation of a population.
        Used for search space analysis and performance evaluation.
        
        Parameters
        ----------
        population : torch.Tensor
            The population to be encoded.
        semantics : torch.Tensor
            The semantics of the population.
        
        Returns
        -------
        torch.Tensor
            The latent representation of the population."""

        self.eval()
        semantic_embedding = self.framework.semantic_embedding(semantics)
        ratio = 0.5
        x = torch.clamp_max(semantic_embedding*ratio+population*(1-ratio),1)
        lat, _ = self.encoder(x)
        return lat

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size).to(x.device)

        _, (hn, cell) = self.lstm(x, (h0, c0))

        return torch.relu(hn), cell

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,
                dropout):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x : torch.tensor, hx : torch.tensor,
                teacher_forcing : bool = False, x_ : torch.tensor = None):
        
        # --- testing novel variation of VAE decoder, where the
        #     decoder receives the latent sampling as input rather 
        #     than as hidden state (not properly autorecursive),
        #     consult MA thesis for why this works better ---
        if teacher_forcing:
            
            x_shifted = torch.cat((x_[:, 0:1, :], x_[:, :-1, :]), dim=1)

            lstm_input = torch.cat((x, x_shifted), dim=-1)

            output, (hx, cell) = self.decoder_lstm(lstm_input)
            prediction = torch.sigmoid(self.fc(output))
            return prediction, (hx, cell)

        else:
            output, (hx, cell) = self.lstm(x, (hx, torch.zeros_like(hx)))
            prediction = torch.sigmoid(self.fc(output))
            return prediction, (hx, cell)
        
        # --- conventional implementation of VAE decoder, make sure to
        #     adjust dimensions before switching back ---
        decoder_sequence = torch.zeros_like(x, dtype=torch.float)
        h, c = hx, torch.randn_like(hx)
        # --- start-of-sequence token <sos> as zeros ---
        token = torch.zeros(x.shape[0], x.shape[2])
        for t in range(decoder_sequence.shape[1]):
            _, (h, c) = self.lstm(token.unsqueeze(1), (h, c))
            #token = torch.softmax(self.fc(h[self.num_layers-1]), dim=1)
            token = torch.sigmoid(self.fc(h[self.num_layers-1]))
            decoder_sequence[:, t, :] = token

            if teacher_forcing:
                token = x[:, t, :]

        return decoder_sequence, (h, c)