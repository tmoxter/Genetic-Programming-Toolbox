import time
import torch
import torch.nn as nn
from joblib.parallel import Parallel, delayed
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from framework import Framework
#import cProfile as profiler
# --> Remove parallization for insightful profiling


class Evolution:
    """
    Simple evolutionary algorithm similar to Genepro that uses the encoding to make small local variations
    in the form of brief random walks in the endoced space. The new variations yield the offspring.
    Selection step is tournament-selection from https://github.com/marcovirgolin/genepro/blob/main/genepro/selection.py.

    Parameters
    ------
    max_evals :int: maximum allowed number of evaluations,
    max_gens :int: maximum allowed number of generations,
    max_time :float: maximum time to run in seconds,
    verbose :bool: per generation statements,
    log :object: torchvision SummaryWriter or None

    Attributes
    -------
    best_of_gen :list: list of best solutions of each generation, pos [-1] for best overall

    """
    
    def __init__(self,
        framework : Framework, model : nn.Module,
        data_x, data_y,
        tournament_size : int = -1,
        population_size : int = 100,
        max_evals : int = None, max_gens : int = 100, max_time : int = None,
        verbose : bool = False, writer : SummaryWriter = None,
        ) -> None:

        self.framework, self.model = framework, model
        self.data_x, self.data_y = data_x, data_y
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.max_evals, self.max_gens = max_evals, max_gens
        self.max_time = max_time
        self.verbose, self.writer = verbose, writer

        self.best_t = list()
        self.best_f = list()
        self.visited = dict()
        self.time_at_gen, self.eval_at_gen = [], []
        self.save_data = []
        self.n_rejected = [0, 0]
        self.num_gens, self.num_evals, self.prev_time = 0, 0, 0.
        
    def evolve(self, proceed : bool = False) -> None:
        """Initialize new generation and evolve until termination criteria are met.
        
        Parameters
        ------
        
        Returns
        ------
        None
        """
        if proceed:
            self.start_time = time.time()-self.prev_time
        if not proceed:
            self.start_time = time.time()
            self.num_gens, self.num_evals = 0, 0
            self._initialize_population()
        
        # --- generational loop ---
        while not self._must_terminate():
            # --- --- perform one generation --- ---
            self._perform_generation()
            # --- --- logging & printing --- ---
            self.save_data.append(
                {
                    "n_gen":self.num_gens, "f":self.fitnesses.max().item(),
                    "fitnesses":self.fitnesses, "n_eval": self.num_evals,
                    "t":time.time() - self.start_time,
                    "tree":self.best_t[-1],
                    #"frac_reject": self.n_rejected[0]/self.n_rejected[1],
                   # "df_gen":torch.tensor(self.df_gen),
                    #"d2_gen":torch.tensor(self.d2_gen),
                    #"df_median":torch.tensor(self.df_gen)
                }
            )
            if self.writer:
                self.writer.add_scalar("Best v. nGen", 
                    self.best_f[-1], self.num_gens)
                self.writer.add_scalar("Median Fitness v. nGen",
                    self.fitnesses.median().item(), self.num_gens)
            if self.verbose:
                print("Gen: {}, evals: {}, best of gen fitness: {:.5f}, gen fitness: {:.3f}"\
                    .format(self.num_gens, self.num_evals, self.best_f[-1],
                    self.fitnesses.sum().item())
                )
    
        self.results = pd.DataFrame.from_dict(self.save_data)
        self.prev_time = time.time() - self.start_time
    
    def _initialize_population(self):
        """Initialize the start population and evaluate its fitness"""
        self.population = self.framework.new_population(self.population_size)
        self.semantics = self.framework.evaluate(self.population)
        self.fitnesses = self.framework.fitness(self.semantics, self.data_y)

    def _perform_generation(self) -> None:
        """Evolve one generation."""

        self.df_gen, self.d2_gen, self.n_rejected = [], [], [0,0]

        # --- variation --- 
        offspring = self.model.variation(self.population, self.semantics)
        assert offspring.shape == self.population.shape,\
            "Model returned missshaped offspring after variation"
        offspring_semantics = self.framework.evaluate(offspring)
        offspring_fitnesses = self.framework.fitness(offspring_semantics,
                                                     self.data_y)
        # --- selection ---
        if self.tournament_size == -1:
            improved = offspring_fitnesses >= self.fitnesses
            self.population[improved] = offspring[improved]
            self.semantics[improved] = offspring_semantics[improved]
            self.fitnesses[improved] = offspring_fitnesses[improved]
        else:
            win_p, win_o = self._tournament_selection(offspring_fitnesses,
                                                      include_parents=True)
            self.population = torch.cat(
                (self.population[win_p], offspring[win_o])
                )
            self.fitnesses = torch.cat(
                (self.fitnesses[win_p], offspring_fitnesses[win_o])
                )
            self.semantics = torch.cat(
                (self.semantics[win_p], offspring_semantics[win_o])
                )
           
        # --- update info ---
        self.num_gens += 1
        
        elitist = torch.argmax(self.fitnesses)
        self.best_t.append(deepcopy(self.population[elitist]))
        self.best_f.append(self.fitnesses[elitist])
        self.time_at_gen.append(time.time() - self.start_time)
        self.eval_at_gen.append(self.num_evals)
    
    def _must_terminate(self) -> bool:
        """Check if termination must occur."""

        self.elapsed_time = time.time() - self.start_time
        if self.max_time and self.elapsed_time >= self.max_time:
            return True
        elif self.max_evals and self.num_evals >= self.max_evals:
            return True
        elif self.max_gens and self.num_gens >= self.max_gens:
            return True
        
        return False
    
    def _tournament_selection(self, offspring_fitnesses : torch.tensor,
                              include_parents : bool = True):

        if include_parents:
            contestants = torch.cat((self.fitnesses, offspring_fitnesses))
        else:
            contestants = offspring_fitnesses

        n_to_select = self.population_size
        n = contestants.shape[0]
        n_per_parse = n // self.tournament_size
        n_parses = n_to_select // n_per_parse

        # --- assert quantities are compatible ---
        assert n / self.tournament_size == n_per_parse,\
            "Number of contestants {} is not a multiple of tournament size {}"\
                                            .format(n, self.tournament_size)
        assert n_to_select / n_per_parse == n_parses
        
        winners = torch.zeros(n_to_select, requires_grad=False, dtype=int)
        indices = torch.arange(0, n, 1)
        for i in range(n_parses):
            p = torch.randperm(n)
            select = torch.argmax(
                contestants[p].reshape((-1, self.tournament_size)),
                axis=1) + torch.arange(0, n, self.tournament_size)
            winners[i*n_per_parse:(i+1)*n_per_parse] = indices[p][select]
        
        if include_parents:
            from_parents = winners[winners < self.population_size]
            from_offspring = winners[winners >= self.population_size]
            return  from_parents, from_offspring % self.population_size
        else:
            return torch.empty(0), winners
       
    def valid_move(self, tree : list):
        """Check if this state has been visited before. Add state to hasmap if new.
        
        Parameters
        --------
        tree :list: to be validated
        
        Returns
        -------
        :bool: whether or not the state is new
        """
        self.n_rejected[1] +=1

        enc = tuple(self.lookup[node.symb] for node in tree[1:])
        if enc in self.visited:
            self.n_rejected[0] +=1
            return False
        else:
            # --- if new state, store fitness ---
            self.visited[enc] = tree[0]
            return True