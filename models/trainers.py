import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from abc import abstractmethod
from itertools import chain
import scipy.special as special

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
    
    def _sammon_loss(self, l1: torch.Tensor, l2: torch.Tensor,
                        y1 : torch.Tensor, y2: torch.Tensor,
                        f1 : torch.Tensor, f2 : torch.Tensor) -> torch.Tensor:
        """Sammon loss for semantic embedding, weighted by fitness
        
        Parameters
        ----------
        l1 : torch.Tensor
            Latent representation of first individual
        l2 : torch.Tensor
            Latent representation of second individual
        y1 : torch.Tensor
            Semantic representation of first individual
        y2 : torch.Tensor
            Semantic representation of second individual
        f1 : torch.Tensor
            Fitness of first individual
        f2 : torch.Tensor
            Fitness of second individual
        
        Returns
        -------
        torch.Tensor"""
        
        y1 = y1[:, 0, :].view(y1.size(0), -1)
        y2 = y2[:, 0, :].view(y2.size(0), -1)
        f1 = f1.nan_to_num(-1e6).clamp(-1e6, 0)
        f2 = f2.nan_to_num(-1e6).clamp(-1e6, 0)
        dy = torch.sum((y1 - y2).abs(), dim=1)
        dx = torch.sum((l1 - l2).abs(), dim=1)
        distances = torch.abs(dx - dy)
        distances = distances.nan_to_num(0).clamp(0, 1e3)
        scale = torch.where(-f1 > -f2, -f1, -f2)
        sammon_loss = torch.abs(distances).div(scale)
        l1_loss = sum(p.abs().sum() for p in self.model.encoder.parameters())
        loss = sammon_loss.mean() + l1_loss*self.l1_coef[0]
        return loss

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

        x_ = self.model(x)
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
        # flatten upper triangular matrices
        dy = dy[torch.triu(torch.ones_like(dy), diagonal=1) == 1].flatten(0)
        de = de[torch.triu(torch.ones_like(de), diagonal=1) == 1].flatten(0)

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
                 learning_rate : tuple, l1_coef : tuple,
                 verbose : bool = False) -> None:
        super().__init__(model, writer, verbose)

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        #self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                           lr=learning_rate[1])
        self.encoder_optimizer = torch.optim.Adam(
            self.model.encoder.parameters(), lr=learning_rate[0]
            )
        self.decoder_optimizer = torch.optim.Adam(
            chain.from_iterable(iter(head.parameters()
                                for head in model.decoder)),
            lr=learning_rate[1]
            )                 
        self.l1_coef = l1_coef

    def train(self, population : torch.Tensor, num_epochs : int,
            batch_size : int, valid_pop : torch.Tensor = None) -> None:
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
        if valid_pop is None:
            valid_pop = population
        try:
            num_epochs = num_epochs[0]
            batch_size = batch_size[0]
        except TypeError:
            pass
        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            
            for data in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                #outputs.squeeze_(0)
                outputs = nn.functional.softmax(outputs, dim = 3)
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
                self.writer.add_scalar("Accuracy/valid", self._multi_head_accuracy(valid_pop),
                                       self.total_epochs+epoch)
        
        self.total_epochs += num_epochs
    
    def train_semantic(self, population : torch.tensor, 
                    semantics : torch.tensor, fitnesses : torch.tensor,
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
        if valid_pop is None:
            valid_pop = population


        # --- use two dataloaders to sample pairs of individuals
        #     if sommon loss is used, otherwise use one dataloader ---
        dataset = SemanticData(population, semantics, fitnesses)
        train_loader_1 = DataLoader(dataset, batch_size)
        train_loader_2 = DataLoader(dataset, batch_size, shuffle=True)

        self.model.train()
        for epoch in range(num_epochs[0]):
            running_loss, correlation = 0.0, 0.0
            
            for sol1 in train_loader_1:
                x1, y1, f1 = sol1
                #x2, y2, f2 = sol2
                # if (x1 == x2).all():
                #     continue
                x1 = x1.view(x1.size(0), -1).type(torch.float)
                #x2 = x2.view(x2.size(0), -1).type(torch.float)
                self.encoder_optimizer.zero_grad()
                latent_1 = self.model.encoder(x1)
                #latent_2 = self.model.encoder(x2)
                #loss = self._sammon_loss(latent_1, latent_2, y1, y2, f1, f2)
                loss, corr = self._correlation_loss(latent_1, y1)
                loss.backward()
                self.encoder_optimizer.step()
                
                running_loss += loss.item()
                correlation += corr.item()
            
            if self.verbose:
                print(f'Epoch [{epoch+1}/{num_epochs[0]}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Correlation/train", correlation/len(train_loader_1),
                                       self.total_epochs+epoch)
        
        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)
        self.model.encoder.requires_grad_(False)
        for epoch in range(num_epochs[0], num_epochs[1]):
            running_loss = 0.0
            
            for data in train_loader:
                self.decoder_optimizer.zero_grad()
                outputs = self.model(data)
                outputs = nn.functional.softmax(outputs, dim = 3)
                data = (data == torch.max(data, dim=2,
                                        keepdim=True).values).float()
                loss = self._multihead_reco_loss(outputs, data,
                                        decoder_l1_only = True)
                loss.backward()
                self.decoder_optimizer.step()
                running_loss += loss.item()
            
            if self.verbose:
                print(f'Epoch [{epoch+1}/{num_epochs[1]}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/valid", self._multi_head_accuracy(valid_pop),
                                       self.total_epochs+epoch)
        
        self.total_epochs = self.total_epochs + num_epochs[0] + num_epochs[1]

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
                learning_rate : tuple, l1_coef : tuple,
                verbose : bool = False) -> None:
        
        super().__init__(model, writer, verbose)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate[0])
        self.l1_coef = l1_coef

    def train(self, population : torch.tensor, num_epochs : int,
              batch_size : int, valid_pop : torch.Tensor = None):
              
        if valid_pop is None:
            valid_pop = population

        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model.train()
        for epoch in range(num_epochs):
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
            
            if self.verbose:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
                self.writer.add_scalar("Accuracy/valid", self.accuracy(valid_pop),
                                       self.total_epochs+epoch)
        
        self.total_epochs += num_epochs
