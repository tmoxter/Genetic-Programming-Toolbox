import torch
import numpy as np
from copy import deepcopy
import math

from framework import Framework

class SyntaxHC:
    """Syntax Hill Climber model. Exhasutively searches the neighbourhood of
    the solutions defined with a hamming distance of 1 and returns the best.
    
    Parameters
    ----------
    framework : Framework
        The framework object.
    target : torch.Tensor
        The target data.
    
    Attributes
    ----------
    variation : callable
        The variation operator for the evolution."""

    def __init__(self, framework : Framework, target : torch.Tensor) -> None:
        self.framework = framework
        self.target = target

    def variation(self, population : torch.tensor, fitnesses : torch.Tensor,
                   *args, **kwargs) -> torch.Tensor:
        """Perform variation.
        
        Parameters
        ----------
        population : torch.tensor
            The population tensor.
        fitnesses : torch.Tensor    
            The fitness tensor.
        
        Returns
        -------
        torch.tensor
            The offspring tensor."""

        n_eval = 0
        tree_len = population.shape[1]
        only_internal = self.framework.treeshape[1] - self.framework.leaf_info[1]
        indices = torch.argmax(population, dim=2).type(torch.float)
        fittest_state = deepcopy(indices)
        backup = deepcopy(indices)
        for j in range(tree_len//2):
            for _ in range(1, self.framework.treeshape[1]):
                           
                indices[:, j] = (indices[:, j] + 1) % self.framework.treeshape[1]
                states = self.framework.syntactic_embedding(
                                        indices.type(torch.long))
                new_fitnesses = self.framework.fitness(
                        self.framework.evaluate(states), self.target)
                improved = new_fitnesses > fitnesses
                fittest_state[improved] = indices[improved]
                fitnesses[improved] = new_fitnesses[improved]
                n_eval += new_fitnesses.shape[0]
            indices = deepcopy(backup)
        for j in range(tree_len//2, tree_len):
            for _ in range(1, self.framework.leaf_info[1]):
                indices[:, j] = (indices[:, j] - only_internal + 1) \
                        % self.framework.leaf_info[1] + only_internal
                states = self.framework.syntactic_embedding(
                                        indices.type(torch.long))
                new_fitnesses = self.framework.fitness(
                        self.framework.evaluate(states), self.target)
                improved = new_fitnesses > fitnesses
                fittest_state[improved] = indices[improved]
                fitnesses[improved] = new_fitnesses[improved]
                n_eval += new_fitnesses.shape[0]
            indices = deepcopy(backup)

        return self.framework.syntactic_embedding(fittest_state.type(torch.long)), n_eval