#-----------------------------------------------#
# Semantic Embedding using Interval Arithmetics #
# ----------------------------------------------#

import torch
from typing import Any
import multiprocessing
from joblib import Parallel, delayed

from framework import Framework

class SeiaGP:

    def __init__(self, framework : Framework) -> None:
        self.framework = framework
        self.n_workes = multiprocessing.cpu_count()
    
    def variation(self, population : Any, semantics : torch.tensor,
                 step_size : float = 1.75) -> torch.tensor:
        
        embedded_space = self.framework.semantic_embedding(semantics)
        angle = torch.rand_like(embedded_space).flatten(1)
        angle = angle.div(torch.norm(angle, p=2, dim=1, keepdim=True))
        embedded_space += angle.reshape_as(embedded_space) * step_size
        
        return self.framework.syntactic_embedding(embedded_space)
    