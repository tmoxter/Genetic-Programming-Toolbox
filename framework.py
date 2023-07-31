import torch
import math
from torch.nn import functional as F
from copy import deepcopy
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import tqdm_joblib

class Framework:
    """Singleton to handle formatting i.e. seamlessy switching between
    1-hot encoding, semantic encoding, trees, readable expressions etc.

    Also function evaluation, fitness, population sampling is implemented.
    
    Some miscellanous functionality like full search space enumeration.
    """

    def __init__(self, x, nodes, max_depth, max_cpu_count :int = 100) -> None:

        n_constants = self._gen_alphabet(nodes, x)
        self.max_depth, self.xdata = max_depth, x
        self.leaf_info = ((2**(max_depth+1)+1)//2, n_constants+x.shape[1])
        self.treeshape = (2**(max_depth+1)-1, len(self.nodes))
        self.n_workers = min(multiprocessing.cpu_count(), max_cpu_count)

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
    
    def resample_repair_population(self, population_size : int,
                    ydata : torch.tensor, lin_scaling : bool = False) -> tuple:
        """Sample population that is adequentely fit (beneficial for initializing
        semantically aware models, without semantic outliers).
        
        Parameters
        ----------
        population_size : int
            The number of individuals in the population.
        ydata : torch.tensor
            The target data.
        
        Returns
        -------
        population : torch.Tensor
            The population.
        semantics : torch.Tensor
            The semantics of the population
        fitness : torch.Tensor
            The fitness of the population
        """
        sample = self.new_population(2*population_size)
        semantics = self.evaluate(sample)
        fitness = self.fitness(semantics, ydata, lin_scaling)
        median = torch.median(fitness)
        keep = fitness >= median
        if keep.sum() > population_size:
            keep = keep.nonzero().flatten()[
                torch.randperm(keep.sum())[:population_size]
            ]
        assert keep.sum() == population_size, "Resampling failure"
        return sample[keep], semantics[keep], fitness[keep]
    
    def _evaluate_atomic_functions(self, indices : torch.tensor,
                                   touched_genes : list = None):
        
        semantics = torch.zeros(indices.shape[0], self.treeshape[0],
                                self.xdata.shape[0], requires_grad=False)
        
        # --- evaluate leaf nodes ---
        for j in range(semantics.shape[0]):
            for i in range(self.treeshape[0]-1, self.treeshape[0]//2-1, -1):
                semantics[j, i, :] = self.nodes[indices[j, i].item()]('','')[:]

        # --- evaluate internal nodes ---
            for i in range(self.treeshape[0]//2-1, -1, -1):
                semantics[j, i, :] = self.nodes[indices[j, i].item()](
                    semantics[j, 2*(i+1)-1, :],semantics[j, 2*(i+1), :])
                                                
        return semantics
    
    @torch.no_grad()
    def evaluate(self, population : torch.tensor, touched_genes : list = None,
                heavy_compute : bool = False):
        indices = torch.argmax(population, dim=2)
        batch = population.shape[0] // self.n_workers +1
        batch = max(batch, 1)
        if heavy_compute:
            with tqdm_joblib(tqdm(desc="Evaluating syntax", total=self.n_workers+2,
                                ascii=False)):
                semantics = torch.vstack(
                    Parallel(n_jobs=self.n_workers)(
                    delayed(self._evaluate_atomic_functions)(
                        indices[i*batch:min((i+1)*batch, indices.shape[0]+1)]
                    )
                    for i in range(self.n_workers+2)
                    )
                )
        else:
            semantics = torch.vstack(
                Parallel(n_jobs=self.n_workers)(
                delayed(self._evaluate_atomic_functions)(
                    indices[i*batch:min((i+1)*batch, indices.shape[0]+1)]
                )
                for i in range(self.n_workers+2)
                )
            )
        return semantics
    
    def syntactic_embedding(self, x : torch.tensor):
        if x.dim() == 1:
            # --- single solution passed ---
            x = x.unsqueeze(0)
        if x.dim() == 2:
            return F.one_hot(x, num_classes=self.treeshape[1])
        if x.dim() == 3:
            internal_idx = torch.argmax(x[:, :self.treeshape[0]//2, :], dim=2)
            leaf_idx = torch.argmax(x[:, self.treeshape[0]//2:,
                                self.treeshape[1]-self.leaf_info[1]:], dim=2)
            indices = torch.cat([
              internal_idx, leaf_idx + self.treeshape[1]-self.leaf_info[1]
            ], dim = 1)
            return F.one_hot(indices, num_classes=self.treeshape[1])

    def _nodewise_semantic(self, current_semantics_: torch.tensor,
                        domains : torch.tensor):
        

        n_samples = 100
        p = torch.ones(current_semantics.size(0))\
                /current_semantics_.size(0)
        idx = p.multinomial(n_samples, replacement=False)
        try:
            current_semantics = current_semantics[idx]
            subset = True
        except IndexError:
            subset = False

        encoded = torch.zeros(self.treeshape[1], requires_grad=False)
        # --- treat leaf nodes ---
        if domains[0] == None:
            epsilon = .1
            for i in range((self.treeshape[1]
                    - self.leaf_info[1]), self.treeshape[1]):
                compare = self.nodes[i](domains[0], domains[1])
                if subset:
                    compare = compare[idx]
                encoded[i] = 1/(1+F.mse_loss(compare,
                            current_semantics)*epsilon)
            
            masked = torch.zeros_like(encoded, requires_grad=False)
            terminals = self.treeshape[1] - self.leaf_info[1]
            masked[-terminals:] = encoded[-terminals:] / encoded[-terminals:].sum()
            return masked

        # --- treat internal nodes ---
        else:
            epsilon = .3
            for i in range(0, self.treeshape[1]):
                if subset:
                    compare = self.nodes[i](domains[0, idx], domains[1, idx])

                encoded[i] = 1/(1+F.mse_loss(compare,
                            current_semantics)*epsilon)
   
            phi = self.treeshape[1] //2
            sort = encoded.argsort(descending=False)[-phi:]
            masked = torch.zeros_like(encoded, requires_grad=False)
            #encoded = encoded / encoded[sort].sum()
            #masked[sort] = F.softmax(encoded[sort], dim=0)
            masked[sort] = encoded[sort] / encoded[sort].sum()
            return masked
    
    def _semantic_batch(self, batch : torch.tensor):
        
        embedding = torch.zeros((batch.shape[0], *self.treeshape),
                                requires_grad=False)
        for j in range(embedding.shape[0]):
            # --- compare leaf functions at leaf node positions ---
            for i in range(self.treeshape[0]-1, self.treeshape[0]//2-1, -1):
                embedding[j, i, :] = self._nodewise_semantic(
                    current_semantics = batch[j, i],
                    domains = (None, None)
                    )
            # --- compare all functions at internal node positions ---
            for i in range(self.treeshape[0]//2-1, -1, -1):
                embedding[j, i, :] = self._nodewise_semantic(
                    current_semantics = batch[j, i],
                    domains = (batch[j, 2*(i+1)-1], batch[j, 2*(i+1)])
                    )
        return embedding

    @torch.no_grad()
    def semantic_embedding(self, semantics : torch.tensor,
                           heavy_compute : bool = False):
        
        batch = semantics.shape[0] // self.n_workers +1
        batch = max(batch, 1)
        if heavy_compute:
            with tqdm_joblib(tqdm(desc="Generating embedding", total=self.n_workers+2,
                                ascii=False)):
                embedding = torch.vstack(
                Parallel(n_jobs=self.n_workers)(
                delayed(self._semantic_batch)(
                    semantics[i*batch:min((i+1)*batch, semantics.shape[0]+1)]
                )
                for i in range(self.n_workers+2)
                )
            )
        else:
            embedding = torch.vstack(
                Parallel(n_jobs=self.n_workers)(
                delayed(self._semantic_batch)(
                    semantics[i*batch:min((i+1)*batch, semantics.shape[0]+1)]
                )
                for i in range(self.n_workers+2)
                )
            )

        return embedding.nan_to_num(0, 1e18, -1e18)
    
    def fitness(self, semantics : torch.tensor, target : torch.tensor,
                lin_scaling : bool = False):
        """Determine fitness score of proram semantics.
        
        Parameters
        ----------
        semantics : torch.tensor
            The semantics of the program.
        target : torch.tensor
            The target values.
        lin_scaling : bool
            Whether apply linear scaling.
        
        Returns
        -------
        :torch.tensor: fitness score of program semantics.
        """
        if lin_scaling:
            y = semantics[:, 0, :]
            se = torch.zeros_like(y)
            for i in range(se.size(0)):
                points = torch.vstack((y[i], target))
                slope = torch.cov(points)[0,1] / (torch.var(y[i]) + 1e-12)
                intercept = torch.mean(target) - slope*torch.mean(y[i])
                se[i] = (slope.item()*y[i] + intercept.item() - target)**2
        else:
            se = (semantics[:, 0, :] - target.repeat(semantics.shape[0], 1))**2
        mse = se.mean(dim = 1)
        mse.nan_to_num_(1e32, 1e32, 1e32)
        return -mse
    
    def classification_fitness(self, semantics : torch.tensor, target : torch.tensor):
        # --- not fully implemented ---
        prediction = torch.sigmoid(semantics[:, 0, :])
        target = target.repeat(semantics.shape[0], 1)
        return (prediction.round() == target).sum(dim=1).float() / target.shape[1]

    def as_tree(self):
        raise NotImplementedError
    
    def print_treelike(self, population : torch.tensor):
        if population.dim() == 2:
            # --- single solution passed ---
            population = population.unsqueeze(0)

        depth = int(math.log2(population.size(1))) + 1
        for tree in population:
            for i in range(depth):
                elems, start = 2**i, 2**i - 1
                end = start + elems
                print(' ' * ((2**(depth - i - 1)) - 1), end='')
                for j in range(start, end):
                    print(tree[j].argmax().item(), end=' ' * (2**(depth-i)-1))
                print()
            print('- '*(population.size(1)//2+1))

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
    
    def hamming_distance(self, population0 : torch.Tensor,
                        population1 : torch.Tensor):
        """
        Calculate the hamming distance between two populations.
        Parameters
        ----------
        population :torch.Tensor
            population of trees

        offspring :torch.Tensor
            offspring of population

        Returns
        -------
        :torch.Tensor: hamming distance between population and offspring
        """
        population_ids = torch.argmax(population0, dim=2)
        offspring_ids = torch.argmax(population1, dim=2)
        return torch.sum(population_ids != offspring_ids, dim=1)
    
    def latent_distance(self, population0 : torch.Tensor, semantics0 : torch.Tensor,
                        population1 : torch.Tensor, semantics1 : torch.Tensor,
                        model : torch.nn.Module):
        """
        Calculate the latent distance between two populations.
        Parameters
        ----------
        population :torch.Tensor
            population of trees

        offspring :torch.Tensor
            offspring of population

        Returns
        -------
        :torch.Tensor: latent distance between population and offspring
        """
        try:
            latent0 = model.latent(population0, semantics0)
            latent1 = model.latent(population1, semantics1)
        except AttributeError:
            return torch.zeros(population0.shape[0])

        return torch.sum((latent0 - latent1)**2, dim=1)**.5
        

    def semantic_distance(self, semantics0 : torch.Tensor,
                                semantics1 : torch.Tensor):
        """
        Calculate the semantic distance between two populations.
        Parameters
        ----------
        population :torch.Tensor
            semantics of population

        offspring :torch.Tensor
            semantics of offspring after variation of population
        
        Returns
        ---------
        :torch.Tensor: semantic distance between population and offspring"""
        return torch.sum((semantics0[:, 0, :]
                            - semantics1[:, 0, :])**2, dim=1)**.5
    
    def fitness_distance(self, fitness0 : torch.Tensor,
                               fitness1 : torch.Tensor):
        """
        Calculate the fitness distance between two populations.
        Parameters
        ----------
        population :torch.Tensor
            fitness of population

        offspring :torch.Tensor
            fitness of offspring after variation of population
        
        Returns
        ---------
        :torch.Tensor: fitness distance between population and offspring"""
        return fitness0 - fitness1
    
    def _gen_alphabet(self, nodes : list, x : torch.tensor):
        """Return the torch callables as well string representations of the
        nodes in the framework alphabet.
        
        Parameters
        ----------
        nodes : list
            The list of nodes to be used
        
        Returns
        -------
        n_constants : int
            The number of constants used
        """

        alphabet_callable = {
            "+": [torch.add],
            "-": [torch.sub],
            "*": [torch.mul],
            "/": [lambda a, b: torch.div(a, b + 1e-9 * (b == 0))],
            "sin": [lambda a, b : torch.sin(a), lambda a, b: torch.sin(b)],
            "cos": [lambda a, b : torch.cos(a), lambda a, b: torch.cos(b)],
            "exp": [lambda a, b : torch.exp(a), lambda a, b: torch.exp(b)],
            "log": [lambda a, b : torch.log(torch.abs(a) + 1e-9 * (a == 0)),
                    lambda a, b : torch.log(torch.abs(b) + 1e-9 * (b == 0))],
            "tan": [lambda a, b : torch.tan(a), lambda a, b: torch.tan(b)],
            "sqrt": [lambda a, b : torch.sqrt(torch.abs(a) + 1e-9 * (a == 0)),
                    lambda a, b : torch.sqrt(torch.abs(b) + 1e-9 * (b == 0))],
            "pow2": [lambda a, b : torch.pow(a, 2), lambda a, b: torch.pow(b, 2)],
            "pow3": [lambda a, b : torch.pow(a, 3), lambda a, b: torch.pow(b, 3)],
            "pow4": [lambda a, b : torch.pow(a, 4), lambda a, b: torch.pow(b, 4)]
        }
        alphabet_str = {
            "+": [lambda a, b : "(%s+%s)"%(a, b)],
            "-": [lambda a, b : "(%s-%s)"%(a, b)],
            "*": [lambda a, b : "(%s*%s)"%(a, b)],
            "/": [lambda a, b : a+"/("+b+")"],
            "sin": [lambda a, b : "sin(%s)"%a, lambda a, b: "sin(%s)"%b],
            "cos": [lambda a, b : "cos(%s)"%a, lambda a, b: "cos(%s)"%b],
            "exp": [lambda a, b : "exp(%s)"%a, lambda a, b: "exp(%s)"%b],
            "log": [lambda a, b : "log(|%s|)"%a, lambda a, b: "log(|%s|)"%b],
            "tan": [lambda a, b : "tan(%s)"%a, lambda a, b: "tan(%s)"%b],
            "sqrt": [lambda a, b : "sqrt(%s)"%a, lambda a, b: "sqrt(%s)"%b],
            "pow2": [lambda a, b : "(%s)^2"%a, lambda a, b: "(%s)^2"%b],
            "pow3": [lambda a, b : "(%s)^3"%a, lambda a, b: "(%s)^3"%b],
            "pow4": [lambda a, b : "(%s)^4"%a, lambda a, b: "(%s)^4"%b]
        }
        # --- add unicode pi if used ---
        constants_str = lambda c: c if c[:4] != "3.14" else '\u03C0'

        used_callabel = list()
        used_str = list()
        n_constants = 0
        for node in nodes:
            if node in alphabet_callable.keys():
                used_callabel += alphabet_callable[node]
                used_str += alphabet_str[node]
            else:
                # --- try to treat as a constant ---
                try:
                    used_callabel.append(
                        lambda a, b, func = node : torch.tensor(
                                            float(func)).repeat(x.shape[0])
                    )
                    used_str.append(
                        lambda a, b, func = node : constants_str(func)
                    )
                    n_constants += 1
                except ValueError:
                    raise ValueError("Node {} not in alphabet".format(node))
        
        used_callabel += [lambda a, b, coef=i: x[:,coef]
                            for i in range(x.shape[1])]
        used_str += [lambda a, b, coef=i: "x_%s"%coef
                            for i in range(x.shape[1])]
        
        self.nodes = {key: val for key, val in enumerate(used_callabel)}
        self.atomic_expr = {key: val for key, val in enumerate(used_str)}
        
        return n_constants
    
    def enumerate_full_space(self, y : torch.Tensor) -> tuple[torch.Tensor,
                                    torch.Tensor,torch.Tensor, torch.Tensor]:
                                                              
        """
        Enumerate the full space of possible trees with the given
        framework parameters.

        Parameters
        ----------
        y : torch.Tensor
            The target values

        Returns
        -------
        :torch.Tensor: full space of possible trees,
        :torch.Tensor: semantics of search space,
        :torch.Tensor: fitness of search space,
        :torch.Tensor: global optima
        """

        size = (self.treeshape[1])**(self.treeshape[0]//2)\
                *self.leaf_info[1]**(self.treeshape[0]//2+1)
        print("Size of configuration space: %s"%size)
        #assert self.treeshape[0] < 8, "Too many nodes to enumerate"

        space_size = self.treeshape[1]**(self.treeshape[0]//2)\
                    *self.leaf_info[1]**(self.treeshape[0]//2+1)
        leaf_min_val = self.treeshape[1] - self.leaf_info[1]
        leaf_min_id = self.treeshape[0]//2+1
        try:
            space_ids = torch.cartesian_prod(*(
                torch.arange(self.treeshape[1]).expand(self.treeshape[0], -1)
                ))
            space_ids = space_ids[
                (space_ids[:, -leaf_min_id:] >= leaf_min_val).all(dim=1)
                ]
        except RuntimeError as e:
            print("# - Configuration space is too large: {%s} - #"%str(e))
            return None, None, None, None
        assert space_ids.shape[0] == space_size, "Space size mismatch"
        self.configspace = self.syntactic_embedding(space_ids)
        self.config_semantics = self.evaluate(self.configspace,
                                            heavy_compute=True)
        self.config_fitness = self.fitness(self.config_semantics, y)
        self.global_optima = self.configspace[self.config_fitness == self.config_fitness.max()]

        return self.configspace, self.config_semantics, \
                self.config_fitness, self.global_optima
    
    def neighborhood_search(self, indices : torch.Tensor, fastfitness : callable):
        """
        Perform a neighborhood search on the given indices.
        Parameters
        ----------
        indices :torch.Tensor
            indices of the current solution
        
        Returns
        -------
        :torch.Tensor: indices of the fittest solution in the neighborhood
        """
        tree_len = indices.shape[0]
        only_internal = self.treeshape[1] - self.leaf_info[1]
        fittest_state = deepcopy(indices)
        var = deepcopy(indices)
        fitnesses = fastfitness(var)
        for j in range(tree_len//2):
            for _ in range(1, self.treeshape[1]):           
                var[j] = (var[j] + 1) % self.treeshape[1]
                new_fitnesses = fastfitness(var)
                improved = new_fitnesses > fitnesses
                if improved:
                    fittest_state, fitnesses = var, new_fitnesses
            var = deepcopy(indices)
        for j in range(tree_len//2, tree_len):
            for _ in range(1, self.leaf_info[1]):
                var[j] = (var[j] - only_internal + 1) \
                        % self.leaf_info[1] + only_internal
                new_fitnesses = fastfitness(var)
                improved = new_fitnesses > fitnesses
                if improved:
                    fittest_state, fitnesses = var, new_fitnesses
            var = deepcopy(indices)

        return fittest_state, (fittest_state == indices).all()
        
    def _hillclimbing(self):
        """
        Map all solutions to their local optima. By hillclimbing with exhaustove
        neighborhood search from every state.
        
        Returns
        -------
        :torch.Tensor: indices of the local optima"""

        fitness_lookup = {tuple(self.configspace[i].argmax(dim=1).tolist()):
                        self.config_fitnesss[i]
                        for i in range(self.configspace.size(0))}
        fastfitness = lambda x: fitness_lookup[tuple(x.tolist())]

        belongs_to_basin = torch.zeros(self.configspace.size(0))
        # --- recursive descent to local optimum ---
        def descent(state):
            new_state, is_locopt = self.neighborhood_search(state, fastfitness)
            return state if is_locopt else descent(new_state)
        # ---
        for i in range(self.configspace.size(0)):
            state = descent(self.configspace[i].argmax(dim=1))
            belongs_to_basin[i] = (
                state.expand(self.configspace.size(0), -1) == self.configspace.argmax(dim=2)
                ).all(dim=1).nonzero()[0]
        
        return belongs_to_basin
            
    def local_optima_network(self):
        """
        Compute the nodes (local optima) and edges (basin transition edges)
        of the local optima network of the problem landscape
        Implementation of Gabriela Ochoa et al. DOI 10.1007/978-3-642-41888-4, ch. 9
        for GP.

        Returns
        -------
        local_optima : list
            list of local optima
        basin_trans_edges : torch.Tensor
            matrix of basin transition edges
        
        """
        try:
            belongs_to_basin = self._hillclimbing()
        except AttributeError:
            self.enumerate_full_space()
            belongs_to_basin = self._hillclimbing()
        
        local_optima = list(set(belongs_to_basin.tolist()))
        solutions_of_basin = {}
        for basin_idx in local_optima:
            solutions_of_basin[int(basin_idx)] = set(
                belongs_to_basin[belongs_to_basin == basin_idx].nonzero().flatten().tolist()
            )
        
        basin_trans_edges = torch.zeros(len(local_optima), len(local_optima))
        for i in range(basin_trans_edges.size(0)):
            basin_i = local_optima[i]
            cardinality = len(solutions_of_basin[basin_i])
            for j in range(basin_trans_edges.size(1)):
                basin_j = local_optima[j]
                pij = 0
                for sol_i in solutions_of_basin[basin_i]:
                    for sol_j in solutions_of_basin[basin_j]:
                        dist = self.hamming_distance(self.configspace[sol_i].unsqueeze(0),
                                                     self.configspace[sol_j].unsqueeze(0))
                        pij += (dist == 1)/self.configspace.size(1)
                basin_trans_edges[i, j] = pij/cardinality
        
        return local_optima, basin_trans_edges
        