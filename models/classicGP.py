import torch
import numpy as np
from copy import deepcopy
import math

from framework import Framework
class ClassicGP:

    def __init__(self, framework : Framework) -> None:
        self.framework = framework

    def variation(self, population : torch.tensor,
                  mutation_chance : float = .1) -> torch.tensor:
        
        offspring = self._crossover(population, uniform_depth=True)
        return self._mutation(offspring, mutation_chance)

    def _crossover(self, population : torch.tensor,
                   uniform_depth : bool = True) -> torch.tensor:
        # --- > Implement test for even population sizes or handle edge case
        pop_size, tree_len = population.shape[:-1]
        depth = int(math.log2(tree_len+1)-1)
        offspring = torch.zeros_like(population)

        parent_pairs = [(i, i + pop_size//2)
                        for i in torch.randperm(pop_size//2)]
        for (parent_a_idx, parent_b_idx) in parent_pairs:
            # --- > implement also Ito's fair-depth subtree selection method,
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
            
            subtree_ids = torch.tensor(subtree_ids)
            remainder_ids = torch.from_numpy(np.setdiff1d(
                np.arange(tree_len), subtree_ids.numpy())
                )
            remainder_ids = torch.tensor(remainder_ids, dtype=int)
            offspring[parent_b_idx, subtree_ids, :] = population[parent_a_idx,
                                                        subtree_ids, :]
            offspring[parent_a_idx, subtree_ids, :] = population[parent_b_idx,
                                                        subtree_ids, :]
            offspring[parent_b_idx, remainder_ids, :] = population[parent_b_idx,
                                                        remainder_ids, :]   
            offspring[parent_a_idx, remainder_ids, :] = population[parent_a_idx,
                                                        remainder_ids, :]
            
        return offspring

    def _mutation(self, population : torch.tensor,
                  chance = float) -> torch.tensor:

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