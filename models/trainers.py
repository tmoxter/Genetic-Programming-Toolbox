import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from abc import abstractmethod
from itertools import chain
import scipy.special as special
from tqdm import tqdm


from models.dataset import PopulationData, SemanticData

class TrainerBase:
    """Base class for all trainers

    Parameters
    ----------
    model : nn.Module
        The model to train
    writer : SummaryWriter
        Tensorboard writer
    verbose : bool, optional
        Whether to print training progress, by default True
    """

    def __init__(self, model : nn.Module, writer : SummaryWriter, verbose : bool = True) -> None:
        self.model, self.writer, self.verbose = model, writer, verbose
        self.total_epochs = 0
    
    @abstractmethod
    def train(self):
        raise NotImplementedError

    def _reconstruction_loss(self, y : torch.tensor, t: torch.tensor
                            ) -> torch.Tensor:
        """Loss for single-head models trained end-to-end or
        for decoder only

        Parameters
        ----------
        y : torch.tensor
            Output of model
        t : torch.tensor
            Target tensor
        
        Returns
        -------
        torch.Tensor """
        
        loss = self.criterion(y, t)
        loss += sum(
            p.abs().sum() for p in self.model.parameters()
            ) * self.l1_coef[1]
        
        return loss.mean()
    
    def _multihead_reco_loss(self, y : torch.tensor, t: torch.tensor,
                             decoder_l1_only : bool = False):
        """Loss for multi-head models trained end-to-end or for decoder only
        
        Parameters
        ----------
        y : torch.tensor
            Output of model
        t : torch.tensor
            Target tensor
        decoder_l1_only : bool, optional
            Whether to only apply L1 regularization to decoder, by default False

        Returns
        -------
        torch.Tensor"""
        
        t = t.expand(y.shape[0], -1, -1, -1)
        p = (1, 3, 0, 2)
        sample_loss = self.criterion(y.permute(*p), t.permute(*p))
        sample_loss = sample_loss.mean(dim = -1)
        n_samples, n_heads = sample_loss.shape
        idx_head = torch.argmin(sample_loss, dim=1)

        loss = sample_loss[:, idx_head].mean()
        l1_loss = 0
        if not decoder_l1_only:
            l1_loss += sum(p.abs().sum()
                        for p in self.model.encoder.parameters())
        l1_sum_heads = torch.tensor([sum(p.abs().sum() for p in head.parameters())
                        for head in self.model.decoder], dtype=torch.float,
                        requires_grad=True)
        l1_rate_heads = idx_head.bincount(minlength=n_heads).type(torch.float)
        l1_loss += l1_sum_heads @ l1_rate_heads / n_samples

        return loss + l1_loss*self.l1_coef[1]

    
    def _weight_decay(self, coef : float, l = 1) -> torch.Tensor:
        """Weight decay for all parameters

        Parameters
        ----------
        weight_decay : float
            Weight decay coefficient
        
        Returns
        -------
        torch.Tensor"""
        
        return sum(p.pow(l).sum() for p in self.model.parameters()) * coef[1]

    @torch.no_grad()
    def _multi_head_accuracy(self, x : torch.Tensor) -> torch.Tensor:
        """
        Node-wise (not tree-wise) accuracy for multi-head models

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
        """
        self.model.eval()
        x_ = self.model(x)
        recon = torch.argmax(x_, dim=3)
        target = torch.argmax(x, dim=2)
        accuracy = (recon == target).type(torch.float).mean(dim=2).T
        accuracy = torch.max(accuracy, dim=1).values
        return accuracy.mean()
    
    @torch.no_grad()
    def accuracy(self, x : torch.Tensor):
        """Node-wise (not tree-wise) accuracy to use when model does not
        support multi-head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor"""
        
        self.model.eval()

        x_, *_ = self.model(x)
        recon = torch.argmax(x_, dim=2)
        target = torch.argmax(x, dim=2)
        acc = (recon == target).type(torch.float).mean()
        return acc
    
    def _correlation_loss(self, x : torch.Tensor, semantics : torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Correlation loss for semantic embedding

        Parameters
        ----------
        x : torch.Tensor
            Latent representation of individual
        semantics : torch.Tensor
            Semantic representation of individual
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]"""

        sem = semantics[:, 0, :].reshape(semantics.size(0), -1)
        xs = x.reshape(x.size(0), -1)
        dy = torch.cdist(sem, sem, p=2)
        de = torch.cdist(xs, xs, p=2)
        # --- flatten upper triangular matrices ---
        dy = dy[torch.triu(torch.ones_like(dy), diagonal=1) == 1].flatten(0)
        de = de[torch.triu(torch.ones_like(de), diagonal=1) == 1].flatten(0)

        dy, de = dy+torch.randn_like(dy)*1e-7, de+torch.randn_like(de)*1e-7

        assert dy.std() != 0 and de.std() != 0, "Zero standard deviation: {}, {}".format(dy.std(), de.std())
        corr = torch.corrcoef(torch.vstack((dy, de)))[0, 1]
        loss = 1 - corr
        l1_loss = sum(p.abs().sum() for p in self.model.encoder.parameters())
        loss += l1_loss*self.l1_coef[0]
        return loss, corr

    @torch.no_grad()
    def correlation(self, x : torch.Tensor, semantics : torch.Tensor):
        """Wrapper for correlation as a metric
        
        Parameters
        ----------
        x : torch.Tensor
            Latent representation of individual
        semantics : torch.Tensor
            Semantic representation of individual
        
        Returns
        -------
        torch.Tensor"""
        
        self.model.eval()
        _, correlation = self._correlation_loss(x, semantics)
        return correlation


class TrainerDenseAE(TrainerBase):
    """Trainer for linear autoencoder. Cross-entropy for recontrsuction loss.
    
    Parameters
    ----------
    model : nn.Module
        The model to train
    writer : SummaryWriter
        Tensorboard writer
    learning_rate : tuple
        Learning rate for encoder and decoder, (encoder, decoder)
    l1_coef : tuple
        L1 regularization coefficient for encoder and decoder
    verbose : bool, optional
        Whether to print training progress, by default True

    """

    def __init__(self, model: nn.Module, writer: SummaryWriter,
                 learning_rate : float, l1_coef : float,
                 verbose : bool = False, validate : bool = False) -> None:
        super().__init__(model, writer, verbose)

        #self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                           lr=learning_rate)
        self.l1_coef = (l1_coef,l1_coef)
        self.validate = validate

        if self.validate:
            self.valid_pop, self.valid_sem, _ = \
                self.model.framework.resample_repair_population(500)
            self.valid_pop = self.model.framework.semantic_embedding(
                self.valid_sem
                )
            
    def train(self, population : torch.Tensor, semantics : torch.Tensor,
            num_epochs : int, batch_size : int, **kwargs) -> None:
        
        """Train the model

        Parameters
        ----------
        population : torch.Tensor
            Population tensor
        num_epochs : int
            Number of epochs to train for
        batch_size : int
            Batch size
        valid_pop : torch.Tensor, optional
            Validation population, by default None
        """
        population = self.model.framework.semantic_embedding(semantics)
        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        for epoch in tqdm(range(num_epochs), desc="Training", ascii=False):
            self.model.train()
            running_loss = 0.0
            
            for data in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                #outputs.squeeze_(0)
                #outputs = nn.functional.softmax(outputs, dim = 3)
                data = (data == torch.max(data, dim=2,
                                        keepdim=True).values).float()
                loss = self._multihead_reco_loss(outputs, data)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            if self.verbose:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/train", self._multi_head_accuracy(population),
                                        self.total_epochs+epoch)
                if self.validate:
                    self.writer.add_scalar("Accuracy/valid", self._multi_head_accuracy(self.valid_pop),
                                       self.total_epochs+epoch)
        
        self.total_epochs += num_epochs
    
    def train_semantic(self, population : torch.Tensor, 
                    semantics : torch.Tensor, fitnesses : torch.Tensor,
                    num_epochs : tuple, batch_size : int,
                    valid_pop : torch.Tensor = None) -> None:
        """Train the model with semantic embedding

        Parameters
        ----------
        population : torch.tensor
            Population tensor
        semantics : torch.tensor
            Semantic tensor
        fitnesses : torch.tensor
            Fitness tensor
        num_epochs : tuple
            Number of epochs to train for, (encoder, decoder)
        batch_size : int
            Batch size
        valid_pop : torch.Tensor, optional
            Validation population, by default None
        """
        
        dataset = SemanticData(population, semantics, fitnesses)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        for epoch in tqdm(range(num_epochs), desc="Training", ascii=False):
            self.model.train()
            running_loss = 0.0
            
            for (data, sem, emb) in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data.float())[0]
                latent = self.model.encoder(data.view(data.size(0), -1).float())
                data = (data == torch.max(data, dim=2,
                                            keepdim=True).values).float()
                recoloss = self._reconstruction_loss(outputs, data)
                corrlos, _ = self._correlation_loss(
                    latent, sem
                    )
                loss = recoloss + corrlos * 2.5e-1
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
    
class TrainerRNNAE(TrainerBase):
    """Trainer for recurrent autoencoder. Cross-entropy for recontrsuction loss.

    Parameters
    ----------
    model : nn.Module
        The model to train
    writer : SummaryWriter
        Tensorboard writer
    learning_rate : tuple
        Learning rate for encoder and decoder, (encoder, decoder)
    l1_coef : tuple
        L1 regularization coefficient for encoder and decoder
    verbose : bool, optional
        Whether to print training progress, by default True
    """

    def __init__(self, model: nn.Module, writer: SummaryWriter,
                learning_rate : float, l1_coef : float,
                semantically : bool = False, verbose : bool = False,
                validate : bool = True) -> None:
        
        super().__init__(model, writer, verbose)

        self.semantically, self.validate = semantically, validate
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate[1])
        self.l1_coef = l1_coef

        if self.validate:
            self.valid_pop, self.valid_sem, _ = \
                self.model.framework.resample_repair_population(500)
            if self.semantically:
                seia = self.model.framework.semantic_embedding(self.valid_sem)
                ratio = 0.5
                self.valid_pop = torch.clamp_max(
                    seia*ratio+self.valid_pop*(1-ratio), 1
                    )

    def train(self, population : torch.Tensor, num_epochs : tuple,
            batch_size : int, semantics : torch.Tensor = None,
            fitnesses : torch.Tensor = None):
            
        if self.semantically:
            self.train_semantic(population, semantics,
                fitnesses, num_epochs, batch_size)
            return
              
        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Training", ascii=False):
            running_loss = 0.0
            
            for data in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data, teacher_forcing = True)
                outputs = nn.functional.softmax(outputs, dim = 2)
                data = (data == torch.max(data, dim=2,
                                            keepdim=True).values).float()
                loss = self._reconstruction_loss(outputs, data)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            # if self.verbose:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/train", self.accuracy(population),
                                        self.total_epochs+epoch)
                if self.validate:
                    self.writer.add_scalar("Accuracy/valid", self.accuracy(self.valid_pop),
                                           self.total_epochs+epoch)
        
        self.total_epochs += num_epochs
    
    def train_semantic(self, population : torch.Tensor, 
                    semantics : torch.Tensor, fitnesses : torch.Tensor,
                    num_epochs : tuple, batch_size : int,
                    valid_pop : torch.Tensor = None) -> None:
        """Train the model with semantic embedding

        Parameters
        ----------
        population : torch.tensor
            Population tensor
        semantics : torch.tensor
            Semantic tensor
        fitnesses : torch.tensor
            Fitness tensor
        num_epochs : tuple
            Number of epochs to train for, (encoder, decoder)
        batch_size : int
            Batch size
        valid_pop : torch.Tensor, optional
            Validation population, by default None
        """

        semantic_embedding = self.model.framework.semantic_embedding(semantics)
        ratio = 0.5
        population = torch.clamp_max(
            semantic_embedding*ratio+population*(1-ratio), 1
            )
        
        dataset = SemanticData(population, semantics, fitnesses)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)
        syntax_reg = int(self.model.encoder.hidden_size * 0.8)

        for epoch in tqdm(range(num_epochs), desc="Training", ascii=False):
            self.model.train()
            running_loss = 0.0
            
            for (data, sem, emb) in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data, teacher_forcing = True)
                #latent = self.model.encoder(emb.type(torch.float))
                latent = self.model.encoder(data.float())
                #outputs = nn.functional.softmax(outputs, dim = 2)
                data = (data == torch.max(data, dim=2,
                                            keepdim=True).values).float()
                recoloss = self._reconstruction_loss(outputs, data)
                corrlos, _ = self._correlation_loss(
                    latent[-1, :, :syntax_reg], sem
                    )
                loss = recoloss + corrlos * 2.5e-1
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            # if self.verbose:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/train", self.accuracy(population),
                                       self.total_epochs+epoch)
                latent = self.model.encoder(population.float())
                self.writer.add_scalar("Correlation/train", self.correlation(latent[-1, :, :syntax_reg],
                                                                semantics),self.total_epochs+epoch)
                if self.validate:
                    self.writer.add_scalar("Accuracy/valid",
                                        self.accuracy(valid_pop),
                                        self.total_epochs+epoch)
                    latent = self.model.encoder(valid_pop.type(torch.float))
                    self.writer.add_scalar("Correlation/valid",
                                           self.correlation(latent[-1, :, :syntax_reg],
                                            self.valid_sem),self.total_epochs+epoch)

        
        self.total_epochs = self.total_epochs + num_epochs


class TrainerRnnVAE(TrainerBase):
    """Trainer for recurrent vriational autoencoder.

    Parameters
    ----------
    model : nn.Module
        The model to train
    writer : SummaryWriter
        Tensorboard writer
    learning_rate : tuple
        Learning rate for encoder and decoder, (encoder, decoder)
    l1_coef : tuple
        L1 regularization coefficient for encoder and decoder
    verbose : bool, optional
        Whether to print training progress, by default True
    """

    def __init__(self, model: nn.Module, writer: SummaryWriter,
                learning_rate : float, l1_coef : float,
                semantically : bool = False, verbose : bool = False,
                validate : torch.Tensor = None) -> None:
        
        super().__init__(model, writer, verbose)

        self.semantically, self.validate = semantically, validate
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate[1])
        self.l1_coef = l1_coef

        if not self.validate is None:
            self.valid_pop = self.validate
            self.valid_sem = self.model.framework.evaluate(self.valid_pop)
            if self.semantically:
                seia = self.model.framework.semantic_embedding(self.valid_sem)
                ratio = 0.5
                self.valid_pop = torch.clamp_max(
                    seia*ratio+self.valid_pop*(1-ratio), 1
                    )
    
    def _elbo(self, recon : torch.Tensor, target : torch.Tensor, mu : torch.Tensor,
            logvar : torch.Tensor) -> torch.Tensor:
        """
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} 
        - \frac{1}{2}
      
        """
    
        kld_weight = 1.25  # Account for the minibatch samples from the dataset
        recons_loss = self.criterion(recon, target)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def train(self, population : torch.Tensor, num_epochs : tuple,
            batch_size : int, semantics : torch.Tensor = None,
            fitnesses : torch.Tensor = None, teacher_forcing : bool = True):
            
        if self.semantically:
            self.train_semantic(population, semantics,
                fitnesses, num_epochs, batch_size, teacher_forcing)
            return
              
        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Training", ascii=False):
            running_loss = 0.0
            
            for data in train_loader:
                self.optimizer.zero_grad()
                outputs, mu, logvar = self.model(data, teacher_forcing = True)
                #outputs = nn.functional.softmax(outputs, dim = 2)
                data = (data == torch.max(data, dim=2,
                                            keepdim=True).values).float()
                loss_dict = self._elbo(outputs, data, mu, logvar)
                loss = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/train", self.accuracy(population),
                                        self.total_epochs+epoch)
                if self.validate:
                    self.writer.add_scalar("Accuracy/valid", self.accuracy(self.valid_pop),
                                           self.total_epochs+epoch)
        
        self.total_epochs += num_epochs
    
    def train_semantic(self, population : torch.Tensor, 
                    semantics : torch.Tensor, fitnesses : torch.Tensor,
                    num_epochs : tuple, batch_size : int,
                    teacher_forcing : bool = False) -> None:
        """Train the model with semantic embedding

        Parameters
        ----------
        population : torch.tensor
            Population tensor
        semantics : torch.tensor
            Semantic tensor
        fitnesses : torch.tensor
            Fitness tensor
        num_epochs : tuple
            Number of epochs to train for, (encoder, decoder)
        batch_size : int
            Batch size
        valid_pop : torch.Tensor, optional
            Validation population, by default None
        """

        semantic_embedding = self.model.framework.semantic_embedding(semantics)
        ratio = 0.5
        population = torch.clamp_max(
            semantic_embedding*ratio+population*(1-ratio), 1
            )
        
        dataset = SemanticData(population, semantics, fitnesses)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        for epoch in tqdm(range(num_epochs), desc="Training", ascii=False):
            self.model.train()
            running_loss = 0.0
            
            for (data, sem, fit) in train_loader:
                self.optimizer.zero_grad()
                outputs, mu, logvar = self.model(data, teacher_forcing = teacher_forcing)
                data = (data == torch.max(data, dim=2,
                                    keepdim=True).values).float()
                loss_dict = self._elbo(outputs, data, mu, logvar)
                corrloss, _ = self._correlation_loss(mu, sem)
                corr_weight = 2e-4
                loss = loss_dict["loss"] + corrloss * corr_weight
                loss += self._weight_decay(self.l1_coef)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/train", self.accuracy(population),
                                       self.total_epochs+epoch)
                outputs, mu, logvar = self.model(population, teacher_forcing = False)
                self.writer.add_scalar("Correlation/train", self.correlation(mu, semantics)
                                                    ,self.total_epochs+epoch)
                self.writer.add_scalar("KLD/train", loss_dict["KLD"], self.total_epochs+epoch)
                self.writer.add_scalar("Mu std/train", mu.std(), self.total_epochs+epoch)
                if not self.validate is None:
                    self.writer.add_scalar("Accuracy/valid",
                                        self.accuracy(self.valid_pop),
                                        self.total_epochs+epoch)
                    outputs, mu, logvar = self.model(self.valid_pop, teacher_forcing = False)
                    self.writer.add_scalar("Correlation/valid",
                                           self.correlation(mu,
                                            self.valid_sem),self.total_epochs+epoch)

        
        self.total_epochs = self.total_epochs + num_epochs


    def train_semantic_fc(self, population : torch.Tensor, 
                    semantics : torch.Tensor, fitnesses : torch.Tensor,
                    num_epochs : tuple, batch_size : int) -> None:
        """Train the model with semantic embedding

        Parameters
        ----------
        population : torch.tensor
            Population tensor
        semantics : torch.tensor
            Semantic tensor
        fitnesses : torch.tensor
            Fitness tensor
        num_epochs : tuple
            Number of epochs to train for, (encoder, decoder)
        batch_size : int
            Batch size
        valid_pop : torch.Tensor, optional
            Validation population, by default None
        """

        semantic_embedding = self.model.framework.semantic_embedding(semantics)
        ratio = 0.5
        population = torch.clamp_max(
            semantic_embedding*ratio+population*(1-ratio), 1
            )
        
        self.model.eval()
        encoded, _ = self.model.encoder(population.type(torch.float))

        dataset = SemanticData(encoded[-1], semantics, fitnesses)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        for epoch in tqdm(range(num_epochs), desc="Training", ascii=False):
            self.model.train()
            running_loss = 0.0
            
            for (data, sem, _) in train_loader:
                self.optimizer.zero_grad()
                mu, logvar = self.model.fc21(data), self.model.fc22(data)
                
                loss_dict = self._elbo(outputs, data, mu, logvar)
                corrloss, _ = self._correlation_loss(mu, sem)
                loss = loss_dict["loss"] + corrloss * 2.5e-3
                loss += self._weight_decay(self.l1_coef)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/train", self.accuracy(population),
                                       self.total_epochs+epoch)
                outputs, mu, logvar = self.model(population, teacher_forcing = True)
                self.writer.add_scalar("Correlation/train", self.correlation(mu, semantics)
                                                    ,self.total_epochs+epoch)
                self.writer.add_scalar("KLD/train", loss_dict["KLD"], self.total_epochs+epoch)
                self.writer.add_scalar("Mu std/train", mu.std(), self.total_epochs+epoch)
                if not self.validate is None:
                    self.writer.add_scalar("Accuracy/valid",
                                        self.accuracy(self.valid_pop),
                                        self.total_epochs+epoch)
                    outputs, mu, logvar = self.model(self.valid_pop, teacher_forcing = True)
                    self.writer.add_scalar("Correlation/valid",
                                           self.correlation(mu,
                                            self.valid_sem),self.total_epochs+epoch)

        
        self.total_epochs = self.total_epochs + num_epochs