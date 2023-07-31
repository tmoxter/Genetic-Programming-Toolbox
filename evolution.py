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
    Evolutionary Algorithm for GP. Variation operators are implemented
    in the model object. Selection is tournament selection or offspring replaces
    parent directly.

    Parameters
    ------
    framework : Framework,
        framework object,
    model : nn.Module
        model object,
    data_x : torch.Tensor
        input data,
    data_y : torch.Tensor
        output data,
    tournament_size : int
        tournament size for tournament selection, -1 for offspring replaces parent,
    trainer : object
        trainer object or None,
    population_size : int
        population size,
    max_evals : int
        maximum allowed number of evaluations,
    max_gens : int
        maximum allowed number of generations,
    max_time : float
        maximum time to run in seconds,
    verbose : bool
        per generation statements,
    writer : SummaryWriter
        torchvision SummaryWriter or None

    Attributes
    -------
    best_of_gen : list
        list of best solutions of each generation, pos [-1] for best overall

    """
    
    def __init__(self,
        framework : Framework, model : nn.Module,
        data_x, data_y, tournament_size : int = -1,
        trainer : object = None, step_size : float = 1,
        population_size : int = 100,
        max_evals : int = None, max_gens : int = 100, max_time : int = None,
        n_epochs : tuple = (800, 200), batch_size : int = 20,
        prevent_revists : bool = False,
        use_scaling : bool = False,
        verbose : bool = False, writer : SummaryWriter = None
        ) -> None:

        self.framework, self.model = framework, model
        self.trainer = trainer
        self.retrain = trainer is not None
        self.data_x, self.data_y = data_x, data_y
        self.step_size = step_size
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.max_evals, self.max_gens = max_evals, max_gens
        self.max_time = max_time
        self.pretrain_epochs, self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose, self.writer = verbose, writer

        self.best_t = list()
        self.best_f = list()
        self.visited = dict()
        self.time_at_gen, self.eval_at_gen = [], []
        self.save_data = []
        self.prevent_revists = prevent_revists
        self.use_scaling = use_scaling
        self.n_rejected = [0, 0]
        self.num_gens, self.num_evals, self.prev_time = 0, 0, 0.
        
    def evolve(self, proceed : bool = False) -> None:
        """
        Initialize new generation and evolve until termination criteria are met.
        
        Parameters
        ------
        proceed :bool: whether or not to continue from previous run
        
        Returns
        ------
        None
        """

        self.num_gens, self.num_evals = 0, 0
        self.hamming, self.latent = 0, 0
        self.delta_sem, self.delta_fit = 0, 0

        if proceed:
            self.start_time = time.time()-self.prev_time
            if self.prevent_revists:
                self._check_state_map(self.population, self.fitnesses,
                                      self.semantics)
        else:
            self.start_time = time.time()
            self._initialize_population()
        
        self._update_data()
        
        # --- generational loop ---
        while not self._must_terminate():
            self._perform_generation()
            self._update_data()

        # --- at end of evolution ---
        self.results = pd.DataFrame.from_dict(self.save_data)
        self.prev_time = time.time() - self.start_time
    
    def _initialize_population(self) -> None:
        """Initialize the start population and evaluate its fitness"""
        self.population, self.semantics, self.fitnesses = \
            self.framework.resample_repair_population(
                    self.population_size, self.data_y, self.use_scaling
                )
        # self.population = self.framework.new_population(self.population_size)
        # self.semantics = self.framework.evaluate(self.population)
        # self.fitnesses = self.framework.fitness(self.semantics, self.data_y,
        #                                         self.use_scaling)
        try:
            # --- save initial model state for iterated local with perturbation ---
            self.model_init = deepcopy(self.model.state_dict())
        except AttributeError:
            self.model_init = None
    
        if self.retrain and self.pretrain_epochs > 0:
            self.trainer.train(self.population, semantics = self.semantics,
                fitnesses = self.fitnesses, num_epochs = self.pretrain_epochs,
                batch_size = self.batch_size)
            
        if self.prevent_revists:
            self._check_state_map(self.population, self.fitnesses,
                                  self.semantics)

    def _perform_generation(self) -> None:
        """Evolve one generation."""

        self.df_gen, self.d2_gen, self.n_rejected = [], [], [0,0]

        # --- training ---
        if self.retrain and self.num_gens > 0:
            # --- reset current model ---
            #self.model.load_state_dict(self.model_init)
            self.trainer.train(self.population, semantics = self.semantics,
                fitnesses = self.fitnesses, num_epochs = self.n_epochs,
                batch_size = self.batch_size)
                    
        # --- variation ---
        offspring = deepcopy(self.population)
        offspring_semantics = deepcopy(self.semantics)
        offspring_fitnesses = deepcopy(self.fitnesses)
        accepted = torch.zeros(self.population_size, dtype=bool)
        attempts = 0
        while not accepted.all():
            ma = ~accepted
            offspring[ma], offspring_semantics[ma], n_eval = self.model.variation(
                population = self.population[ma], semantics = self.semantics[ma],
                fitnesses = self.fitnesses[ma], ydata = self.data_y,
                mutation_chance = 1/self.population.shape[1],
                step_size = self.step_size * 1.01**attempts,
                operator = self.model.operator)

            #offspring_semantics = self.framework.evaluate(offspring)
            offspring_fitnesses[ma] = self.framework.fitness(offspring_semantics[ma],
                                                self.data_y, self.use_scaling)
               
            if self.prevent_revists:
                accepted += ~self._check_state_map(offspring, offspring_fitnesses,
                                        offspring_semantics)
                if self.writer:
                    self.writer.add_scalar("Discovered",
                            len(self.visited.keys()), self.num_gens)
            else:
                accepted += torch.ones(offspring.shape[0], dtype=bool)
            attempts+=1
            if attempts == 1000:
                break
        # --- ---
        assert offspring.shape == self.population.shape and\
            offspring_fitnesses.shape == self.fitnesses.shape and \
            offspring_semantics.shape == self.semantics.shape,\
            "Model returned missshaped offspring after variation: {} {} {}"\
                .format(offspring.shape, offspring_fitnesses.shape,
                        offspring_semantics.shape)

        self.hamming = self.framework.hamming_distance(
                        offspring, self.population)           
        try:
            self.latent = self.framework.latent_distance(
                offspring, offspring_semantics, self.population,
                self.semantics, self.model)      
        except AttributeError:
            self.latent = 0
        self.delta_sem = self.framework.semantic_distance(
                            offspring_semantics, self.semantics)
        self.delta_fit = self.framework.fitness_distance(
                            offspring_fitnesses, self.fitnesses)
        
        if self.writer:
            joint = torch.cat((offspring, self.population), dim=0)
            new = (joint.argmax(dim=2).unique(dim=0).shape[0] 
                - self.population.argmax(dim=2).unique(dim=0).shape[0])\
                /offspring.shape[0]
            self.writer.add_scalar("Novel", new, self.num_gens)
        
        # --- selection ---
        if self.tournament_size == -1:
            improved = offspring_fitnesses >= self.fitnesses
            self.population[improved] = offspring[improved]
            self.semantics[improved] = offspring_semantics[improved]
            self.fitnesses[improved] = offspring_fitnesses[improved]
            n_eval += self.population_size
        else:
            win_p, win_o = self._tournament_selection(offspring_fitnesses,
                                                    include_parents = True)
            self.population = torch.cat(
                (self.population[win_p], offspring[win_o])
                )
            self.fitnesses = torch.cat(
                (self.fitnesses[win_p], offspring_fitnesses[win_o])
                )
            self.semantics = torch.cat(
                (self.semantics[win_p], offspring_semantics[win_o])
                )
            n_eval += self.population_size
        
        self.num_gens += 1
        self.num_evals += n_eval
    
    def _must_terminate(self) -> bool:
        """Check if termination must occur."""

        self.elapsed_time = time.time() - self.start_time
        if self.max_time and self.elapsed_time >= self.max_time:
            return True
        elif self.max_evals and self.num_evals >= self.max_evals:
            return True
        elif self.max_gens and self.num_gens >= self.max_gens:
            return True
        elif self.best_f[-1] == 0: # -> add attribute for best fitness if known
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
    
    def _update_data(self):
    
        elitist = torch.argmax(self.fitnesses)
        self.best_t.append(deepcopy(self.population[elitist]))
        self.best_f.append(self.fitnesses[elitist])
        self.time_at_gen.append(time.time() - self.start_time)
        self.eval_at_gen.append(self.num_evals)

        self.save_data.append(
                {
                "n_gen":self.num_gens, "f":self.fitnesses.max().item(),
                "n_eval": self.num_evals,
                "time":time.time() - self.start_time,
                "elitist":self.best_t[-1],
                "frac_reject": self.n_rejected[0]/self.n_rejected[1]\
                    if self.prevent_revists else 0,
                "hamming":self.hamming, "latent":self.latent,
                "delta_sem":self.delta_sem, "delta_fit":self.delta_fit,
                "population": deepcopy(self.population),
                "pop_fitness":deepcopy(self.fitnesses),
                }
            )
        
        if self.writer:
            self.writer.add_scalar("Best v. nGen",
                self.best_f[-1], self.num_gens)
            self.writer.add_scalar("Median Fitness v. nGen",
                self.fitnesses.median().item(), self.num_gens)
            self.writer.add_scalar("Mean Fitness v. nGen",
                self.fitnesses.mean().item(), self.num_gens)
            if self.num_gens > 0:
                self.writer.add_scalar("Mean Hamming Distance v. nGen",
                        self.hamming.float().mean().item(), self.num_gens)
                self.writer.add_scalar("Median semantic Distance v. nGen",
                            self.delta_sem.median().item(), self.num_gens)
        if self.verbose:
            print("Gen: {}, Evals: {}\nPopulation fitness: {:.2f}, Best of gen: {:.5f}"\
                    .format(self.num_gens, self.num_evals,
                    self.fitnesses.sum().item(), self.best_f[-1])
                )
            print(' ')
       
    def _check_state_map(self, offspring : torch.Tensor, offspring_fitness : torch.Tensor,
                   offspring_semantics : torch.Tensor):
        """Check if this state has been visited before. Add state to hashmap if new.
        
        Parameters
        --------
        offspring :torch.Tensot
            to be validated and added to hashmap
        
        ofsspring_fitness :torch.Tensor
            fitness of offspring
        
        offspring_semantics :torch.Tensor
            semantics of offspring
        
        Returns
        -------
        :bool: whether or not the state is new
        """
        self.n_rejected[1] += offspring.shape[0]

        offspring_ids = torch.argmax(offspring, dim=2)
        reject = torch.zeros(offspring_ids.shape[0], dtype=bool)

        for i in range(offspring_ids.shape[0]):
            enc = tuple(offspring_ids[i].tolist())

            if enc in self.visited:
                self.n_rejected[0] +=1
                reject[i] = True
            else:
                # --- if new state, store fitness, semantics and latent rep ---
                self.visited[enc] = {"fitness":offspring_fitness[i],
                                     "semantics":offspring_semantics[i]}
        return reject