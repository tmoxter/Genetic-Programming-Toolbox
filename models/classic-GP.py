import torch
from torch import random
class ClassicGP:

    def __init__(self) -> None:
        pass

    def variation(self, population : torch.tensor,
                  mutation_chance : float = .1) -> torch.tensor:
        pass

    def _crossover(self, population : torch.tensor) -> torch.tensor:
        pop_size, tree_len = population.shape[0, 1]
        depth = torch.log2(tree_len+1)-1
        parent_a = population[:pop_size//2].clone()
        parent_b = population[pop_size//2:].clone()
        offspring = torch.zeros_like(population)

        parent_pairs = [(i, i + pop_size//2) for i in torch.randperm(pop_size//2)]

        for (parent_a_idx, parent_b_idx) in parent_pairs:
            st_root = random.randint(0, tree_len)
    
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
            
            remainder_ids = set(range(tree_len)) - set(subtree_ids)
            subtree_ids = torch.tensor(subtree_ids)
            remainder_ids = torch.tensor(remainder_ids)
            offspring[parent_b_idx][subtree_ids, :] = parent_a[parent_a_idx][subtree_ids, :]
            offspring[parent_a_idx][subtree_ids, :] = parent_b[parent_b_idx][subtree_ids, :]
            offspring[parent_b_idx][remainder_ids, :] = parent_a[parent_b_idx][remainder_ids, :]
            offspring[parent_a_idx][remainder_ids, :] = parent_b[parent_a_idx][remainder_ids, :]
            
        return offspring


    def _mutation(self, population : torch.tensor) -> torch.tensor:
        pass