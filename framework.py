import torch
from torch.nn import functional as F
from torch import finfo
import multiprocessing
from joblib import Parallel, delayed

class Framework:

    def __init__(self, x, nodes, n_leafnodes, max_depth) -> None:
        
        nodes = [torch.mul, torch.add,
                 lambda a, b: torch.div(a, b + 1e-9 * (b == 0)),
                 torch.sub,
                 
                 lambda a, b : torch.sin(a), lambda a, b: torch.sin(b),
                 lambda a, b : torch.cos(a), lambda a, b: torch.cos(b),
                 lambda a, b : torch.exp(a), lambda a, b: torch.exp(b),
                 lambda a, b : torch.log(torch.abs(a) + 1e-9 * (a == 0)),
                 lambda a, b : torch.log(torch.abs(b) + 1e-9 * (b == 0)),

                 lambda a, b : torch.tensor(5).repeat(x.shape[0]),
                 lambda a, b : torch.tensor(3.141).repeat(x.shape[0])]
        
        nodes += [lambda a, b : x[:,i] for i in range(x.shape[1])]

        atomic_expr = [
                lambda a, b : a+"*"+b, lambda a, b: "("+a+"+"+b+")",
                lambda a, b : a+"/("+b+")", lambda a, b: "("+a+"-"+b+")",
                lambda a, b : "sin(%s)"%a, lambda a, b: "sin(%s)"%b,
                lambda a, b :"cos(%s)"%a, lambda a, b: "cos(%s)"%b,
                lambda a, b : "exp(%s)"%a, lambda a, b: "exp(%s)"%b,
                lambda a, b : "log(|%s|)"%a, lambda a, b: "log(|%s|)"%b,

                lambda a, b : "5",
                lambda a, b : '\u03C0'
        ]
        atomic_expr += [lambda a, b : "x_%s"%i for i in range(x.shape[1])]

        self.nodes = {key: val for key, val in enumerate(nodes)}
        self.atomic_expr = {key: val for key, val in enumerate(atomic_expr)}
        self.max_depth, self.xdata = max_depth, x
        self.leaf_info = ((2**(max_depth+1)+1)//2, n_leafnodes)
        self.treeshape = (2**(max_depth+1)-1, len(nodes))
        self.n_workers = 1#multiprocessing.cpu_count()

    def new_population(self, population_size : int):
        sample_intern = torch.randint(0, self.treeshape[1]-self.leaf_info[1],
                (population_size, self.treeshape[0]-self.leaf_info[0]))
        sample_leaf = torch.randint(self.treeshape[1]-self.leaf_info[1],
                self.treeshape[1], (population_size, self.leaf_info[0]))
        sample = torch.cat((sample_intern, sample_leaf), dim=1)
        return self.syntactic_embedding(sample)

    def new_subtree(self, max_depth : int):
        treeshape = (2**(max_depth+1)-1, len(self.nodes))
        sample = torch.randint(0, treeshape[1],(1, treeshape[0]))
        return self.syntactic_embedding(sample)
    
    def _evaluate_atomic_functions(self, indices : torch.tensor, shape : tuple,
                                  alphabet : dict, touched_genes : list = None):
        
        semantics = torch.zeros(indices.shape[0], shape[0], shape[1],
                                requires_grad=False)
        
        # --- evaluate leaf nodes ---
        for j in range(semantics.shape[0]):
            for i in range(shape[0]-1, shape[0]//2-1, -1):
                semantics[j, i, :] = alphabet[indices[j, i].item()]('','')[:]

        # --- evaluate internal nodes ---
            for i in range(shape[0]//2-1, -1, -1):
                semantics[j, i, :] = alphabet[indices[j, i].item()](
                    semantics[j, 2*(i+1)-1, :],semantics[j, 2*(i+1), :])
                                                
        return semantics
    
    @torch.no_grad()
    def evaluate(self, population : torch.tensor):
        indices = torch.argmax(population, dim=2)
        semantics = torch.vstack(
            Parallel(n_jobs=self.n_workers)(
            delayed(self._evaluate_atomic_functions)(
            indices[i::self.n_workers],
            shape = (self.treeshape[0], self.xdata.shape[0]),
            alphabet = self.nodes
            )
            for i in range(self.n_workers)
        ))

        return semantics
    
    def syntactic_embedding(self, x : torch.tensor):
        if x.dim() == 1:
            # --- single solution passed ---
            x = x.unsqueeze(0)
        if x.dim() == 2:
            return F.one_hot(x, num_classes=self.treeshape[1])
        if x.dim() == 3:
            indices = torch.argmax(x, dim=2)
            return F.one_hot(indices, num_classes=self.treeshape[1])

    def _nodewise_semantic(self, current_semantics: torch.tensor,
                        domains : torch.tensor):
        
        encoded = torch.zeros(self.treeshape[1], requires_grad=False)
        case_leaf = int((domains[0] == None)) * (self.treeshape[1] - self.leaf_info[1])
        for i in range(case_leaf, self.treeshape[1]):
            compare = self.nodes[i](domains[0], domains[1])
            encoded[i] = 1/(1+F.mse_loss(compare, current_semantics)*.2)

        sort = encoded.argsort(descending=False)[-5:]
        masked = torch.zeros_like(encoded, requires_grad=False)
        #encoded = encoded / encoded[sort].sum()
        #masked[sort] = F.softmax(encoded[sort], dim=0)
        masked[sort] = encoded[sort] / encoded[sort].sum()
        return masked

    @torch.no_grad()
    def semantic_embedding(self, semantics : torch.tensor):
        
        semantics.clamp_(-1e18, 1e18)
        embedding = torch.zeros((semantics.shape[0], *self.treeshape),
                                requires_grad=False)
        for j in range(embedding.shape[0]):
            # --- compare leaf functions at leaf node positions ---
            for i in range(self.treeshape[0]-1, self.treeshape[0]//2-1, -1):
                embedding[j, i, :] = self._nodewise_semantic(
                    current_semantics = semantics[j, i],
                    domains = (None, None)
                    )
            # --- compare all functions at internal node positions ---
            for i in range(self.treeshape[0]//2-1, -1, -1):
                embedding[j, i, :] = self._nodewise_semantic(
                    current_semantics = semantics[j, i],
                    domains = (semantics[j, 2*(i+1)-1], semantics[j, 2*(i+1)])
                    )
        return embedding
    
    def fitness(self, semantics : torch.tensor, target : torch.tensor):

        se = (semantics[:, 0, :] - target.repeat(semantics.shape[0], 1))**2
        return se.mean(dim = 1)

    def as_tree(self):
        raise NotImplementedError

    def as_expression(self, population : torch.tensor):
        if population.dim() == 2:
            # --- single solution passed ---
            population = population.unsqueeze(0)
        indices = torch.argmax(population, dim=2)
        output = list()

        for j in range(indices.shape[0]):
            string = ['']*self.treeshape[0]
            # --- stringify leaf nodes ---
            for i in range(self.treeshape[0]-1, self.treeshape[0]//2-1, -1):
                string[i] = self.atomic_expr[indices[j, i].item()]('','')[:]
            # --- stringify interanl nodes ---
            for i in range(self.treeshape[0]//2-1, -1, -1):
                string[i] = self.atomic_expr[indices[j, i].item()](
                    string[2*(i+1)-1], string[2*(i+1)])
            output.append(string[0])
        return output
        