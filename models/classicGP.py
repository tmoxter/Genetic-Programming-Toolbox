import torch
import numpy as np
from copy import deepcopy
import math

from framework import Framework

class ClassicGP:
    """
    Classic GP model to perform standard gp variation operators
    'crossover', 'subtree-mutation', 'node-mutation'.

    Parameters
    ----------
    framework : Framework-object
        The framework object,
    operator : str, ['mixed' (i.e. crossover & subtree mut.), 'subtree-mutation', 'node-mutation']
            The variation operator to be used, by default "subtree-mutation"
    """

    def __init__(self, framework : Framework,
                 operator : str = "subtree-mutation"):
        self.framework = framework
        self.operator = operator

    def variation(self, population : torch.tensor,
                mutation_chance : float = 0.1,
                *args, **kwargs) -> torch.tensor:
        """
        Parameters
        ----------
        population : torch.tensor
            The population to be varied.
        mutation_chance : float, optional
            The chance of mutation, by default 0.1
        
        Returns
        -------
        torch.tensor
            The varied population.
        """
        if self.operator == "mixed":
            offspring = self._crossover(population, uniform_depth=True)
            # --- --- muatation chance is per node --- ---
            offspring = self._subtree_mutation(offspring, mutation_chance)
            offspring_semantics = self.framework.evaluate(offspring)
            return offspring, offspring_semantics, 0
        if self.operator == "subtree-mutation":
            # --- --- muatation chance is per tree (=1) --- ---
            offspring = self._subtree_mutation(population, 1)
            offspring_semantics = self.framework.evaluate(offspring)
            return offspring, offspring_semantics, 0
        if self.operator == "node-mutation":
            # --- --- unsed as variation operator --- ---
            #( --- --- muatation chance is per node --- --- )
            offspring = self._node_mutation(population)
            offspring_semantics = self.framework.evaluate(offspring)
            return offspring, offspring_semantics, 0
        
        raise NotImplementedError("Operator {} not implemented. Choose from ".format(self.operator)\
                                  +"'mixed','subtree-mutation', 'node-mutation'"
                                  )

    def _crossover(self, population : torch.tensor,
                   uniform_depth : bool = True) -> torch.tensor:
        # --- check if population size is even ---
        assert population.shape[0] % 2 == 0, "Population size must be even."
        pop_size, tree_len = population.shape[:-1]
        depth = int(math.log2(tree_len+1)-1)
        offspring = torch.zeros_like(population)

        parent_pairs = [(i, i + pop_size//2)
                        for i in torch.randperm(pop_size//2)]
        for (parent_a_idx, parent_b_idx) in parent_pairs:
            # --- > implement also Ito's fair-depth subtree selection method?
            #       however equal probability sampling has been shown to be more effective in RDO
            if uniform_depth:
                sample_depth = torch.randint(depth+1, (1,)).item()
                if sample_depth == 0:
                    st_root = sample_depth
                else:
                    st_root = torch.randint(2**sample_depth-1,
                                            2**(sample_depth+1)-1,(1,)).item()
            else:
                st_root = torch.randint(tree_len, (1,)).item()
    
            subtree_ids = [st_root]
            next_lvl = [st_root]
            for _ in range(depth):
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
            
            subtree_ids = torch.tensor(subtree_ids)
            remainder_ids = torch.from_numpy(np.setdiff1d(
                np.arange(tree_len), subtree_ids.numpy())
                ).type(torch.long)
            offspring[parent_b_idx, subtree_ids, :] = population[parent_a_idx,
                                                        subtree_ids, :]
            offspring[parent_a_idx, subtree_ids, :] = population[parent_b_idx,
                                                        subtree_ids, :]
            offspring[parent_b_idx, remainder_ids, :] = population[parent_b_idx,
                                                        remainder_ids, :]   
            offspring[parent_a_idx, remainder_ids, :] = population[parent_a_idx,
                                                        remainder_ids, :]
            
        return offspring

    def _mutation(self, population : torch.Tensor,
                  chance : float = 1) -> torch.Tensor:

        tree_len = population.shape[1]
        indices = torch.argmax(population, dim=2).type(torch.float)
        mask = torch.rand_like(indices) < chance
        internal, leaves = torch.zeros_like(mask), torch.zeros_like(mask)
        internal[:, :tree_len//2] = mask[:, :tree_len//2]
        leaves[:, tree_len//2:] = mask[:, tree_len//2:]
        indices[internal] = torch.randint_like(indices, 0,
                                            population.shape[2])[internal]
        indices[leaves] = torch.randint_like(indices,
                        population.shape[2] - self.framework.leaf_info[1],
                        population.shape[2])[leaves]

        return self.framework.syntactic_embedding(indices.type(torch.long))
    
    def _subtree_mutation(self, population : torch.Tensor,
        uniform_depth : bool = True, chance : float = 1) -> torch.Tensor:
    
    
        tree_len = population.shape[1]
        depth = int(math.log2(tree_len+1)-1)
        offspring = deepcopy(population).type(torch.long)
        selected = torch.rand(population.size(0)) < chance
        n_selected = torch.sum(selected)
        st_root = torch.zeros(population.size(0), dtype=torch.long)
        
        if uniform_depth:
            sample_depth = torch.randint(depth+1, (n_selected,))
            scale = ((2**(sample_depth+1)-1) - (2**sample_depth-1))
            st_root[selected] = (2**sample_depth-1
                    + torch.rand((n_selected,)) * scale).type(torch.long)
                                            
        else:
            st_root[selected] = torch.randint(tree_len, (n_selected,))
            
        for i in torch.arange(population.size(0))[selected]:
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
            offspring[i, subtree_ids, :] = self._mutation(
                population[i].unsqueeze(0), chance=1
                )[0, subtree_ids, :]
            
        return offspring

    def _node_mutation(self, population : torch.Tensor):
        """Uniformly sample node and replace with random node."""

        selected = torch.randint(population.size(1), (population.size(0),))
        offspring = deepcopy(population)
        offspring[torch.arange(population.size(0)), selected, :] = self._mutation(
                population, chance=1
                )[torch.arange(population.size(0)), selected, :]

        return offspring
    
    def _protect(self, offspring : torch.Tensor, population : torch.Tensor):
        # --- --- syntactic protection and rejection --- ---                            
        leaves = offspring.argmax(dim=2)[:, -self.framework.leaf_info[0]:]
        leaf_primitive = self.framework.treeshape[1] - self.framework.leaf_info[1]
        refuse = (leaves < leaf_primitive).any(dim=1)
        offspring[refuse] = population[refuse]
        return offspring
        
        