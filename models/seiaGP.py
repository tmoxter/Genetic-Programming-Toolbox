#-----------------------------------------------#
# Semantic Embedding using Interval Arithmetics #
# ----------------------------------------------#

import torch
from copy import deepcopy
import math

from framework import Framework

class SeiaGP:

    def __init__(self, framework : Framework) -> None:
        self.framework = framework
    
    def variation(self, population : torch.Tensor, semantics : torch.Tensor,
                 step_size : float = 1, operator : str = "subtree-mutation",
                 *args, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        population : torch.Tensor
            The population to be varied.
        semantics : torch.Tensor
            The semantics of the population.
        step_size : float, optional
            The step size for the variation operator, by default 1
        operator : str, ['crossover', 'subtree-mutation', 'node-mutation']
            The variation operator to be used, by default "subtree-mutation"
        
        Returns
        -------
        torch.Tensor
            The varied population.
        """
        
        if operator == "closest":
            return self._force_closest(population, semantics)
        if operator == "subtree-mutation":
            return self._subtree_mutation(population, semantics,
                                          step_size=step_size)
        if operator == "node-mutation":
            return self._node_mutation(population, semantics)
       
        raise NotImplementedError("Operator {} not implemented."
                                  .format(operator))
    
    def _force_closest(self, population : torch.Tensor,
                       semantics : torch.Tensor) -> torch.Tensor:
        """Change single node to overall (semantically) most similar alternative."""
        
        seia = self.framework.semantic_embedding(semantics)
        indices = torch.argmax(population, dim=2).type(torch.long)
        maxmask = seia == torch.max(seia, dim=2, keepdim=True)[0]
        seia[maxmask] = 0
        selected = torch.argmax(seia.max(dim=2)[0], dim=1)
        indices[torch.arange(population.size(0)), selected] = \
            seia[torch.arange(population.size(0)), selected, :].argmax(dim=1)
        
        return self.framework.syntactic_embedding(indices), \
        population.size(0) * population.size(1)/2 * population.size(2)
    
    def _node_mutation(self, population : torch.Tensor,
                       semantics : torch.Tensor) -> torch.Tensor:
        """Change single node to similar alternative by treating SEIA of node as
        discrete probabilities."""

        seia = self.framework.semantic_embedding(semantics)
        indices = torch.argmax(population, dim=2).type(torch.long)
        selected = torch.randint(population.size(1), (population.size(0),))
        maxmask = seia == torch.max(seia, dim=2, keepdim=True)[0]
        seia[maxmask] = 0

        indices[torch.arange(population.size(0)), selected] = \
            seia[torch.arange(population.size(0)), selected, :].multinomial(1).squeeze()
        
        return self.framework.syntactic_embedding(indices), \
            population.size(0) * self.framework.treeshape[1]/2
    
    def _subtree_mutation(self, population : torch.Tensor, semantics : torch.Tensor,
        uniform_depth : bool = True, step_size : float = 1) -> torch.Tensor:
        """Sample subtree, perform local pertubation in (flat) SEIA space"""

        seia = self.framework.semantic_embedding(semantics)
                
        tree_len = population.shape[1]
        depth = int(math.log2(tree_len+1)-1)
        offspring = deepcopy(population).type(torch.long)
        n = population.size(0)
        st_root = torch.zeros(population.size(0), dtype=torch.long)
        
        if uniform_depth:
            sample_depth = torch.randint(depth+1, (n,))
            scale = ((2**(sample_depth+1)-1) - (2**sample_depth-1))
            st_root = (2**sample_depth-1
                    + torch.rand((n,)) * scale).type(torch.long)                       
        else:
            st_root = torch.randint(tree_len, (n,))

        n_eval = 0
        for i in torch.arange(population.size(0)):
            subtree_ids = [st_root[i].item()]
            next_lvl = [st_root[i].item()]
            for level in range(depth):
                current_lvl = next_lvl
                next_lvl = []
                for node_idx in current_lvl:
                    child1 = 2 * (node_idx + 1) - 1
                    child2 = 2 * (node_idx + 1)
                    if child1 < 2 ** (depth + 1) - 1:
                        subtree_ids.append(child1)
                        next_lvl.append(child1)
                    if child2 < 2 ** (depth + 1) - 1:
                        subtree_ids.append(child2)
                        next_lvl.append(child2)
                    
            subtree_ids = torch.tensor(subtree_ids).type(torch.long)
            subtree = seia[i, subtree_ids, :]
            angle = torch.rand_like(subtree).flatten(1)
            angle = angle.div(torch.norm(angle, p=2, dim=1, keepdim=True))
            subtree += angle.reshape_as(subtree) * step_size
            subtree[subtree_ids > tree_len//2 ] *= torch.cat(
                (torch.zeros(population.size(2)- self.framework.leaf_info[1]),
                torch.ones(self.framework.leaf_info[1]))
            )
            offspring[i, subtree_ids] = self.framework.syntactic_embedding(subtree.unsqueeze(0))
            n_eval += subtree_ids.size(0) * self.framework.treeshape[1]/2
        
        return offspring, n_eval
