import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from abc import abstractmethod

from models.dataset import PopulationData

class TrainerBase:

    def __init__(self, model : nn.Module, writer : SummaryWriter) -> None:
        self.model, self.writer = model, writer
        self.total_epochs = 0
    
    @abstractmethod
    def train(self):
        """
        """
        raise NotImplementedError

    def _reconstruction_loss(self, y : torch.tensor, t: torch.tensor):
        loss = self.criterion(y, t)
        loss += sum(
            p.abs().sum() for p in self.model.parameters()
            ) * self.l1_coef
        
        return loss
    
    @torch.no_grad()
    def accuracy(self, x : torch.tensor):
        """
        Note-wise (not tree-wise) accuracy
        """
        self.model.eval()

        x_ = self.model(x)
        recon = torch.argmax(x_, dim=2)
        target = torch.argmax(x, dim=2)
        return (recon == target).type(torch.float).mean()

class TrainerDenseAE(TrainerBase):

    def __init__(self, model: nn.Module, writer: SummaryWriter,
                 learning_rate : float, l1_coef : float) -> None:
        super().__init__(model, writer)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.l1_coef = l1_coef

    def train(self, population : torch.tensor, num_epochs : int,
              batch_size : int):

        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for data in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                outputs = nn.functional.softmax(outputs, dim = 2)
                data = (data == torch.max(data, dim=2,
                                            keepdim=True).values).float()
                loss = self._reconstruction_loss(outputs, data)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
        
        self.total_epochs += num_epochs

class TrainerRNNAE(TrainerBase):

    def __init__(self, model: nn.Module, writer: SummaryWriter,
                learning_rate : float, l1_coef : float) -> None:
        
        super().__init__(model, writer)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.l1_coef = l1_coef

    def train(self, population : torch.tensor, num_epochs : int,
              batch_size : int):

        dataset = PopulationData(population)
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for data in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                outputs = nn.functional.softmax(outputs, dim = 2)
                data = (data == torch.max(data, dim=2,
                                            keepdim=True).values).float()
                loss = self._reconstruction_loss(outputs, data)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
            if self.writer:
                self.writer.add_scalar("Loss/train", running_loss, self.total_epochs+epoch)
        
        self.total_epochs += num_epochs
