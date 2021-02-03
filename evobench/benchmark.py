from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from multiprocessing import Manager, Pool, RLock
from typing import Dict

import numpy as np
from lazy import lazy
from tqdm.auto import tqdm

from evobench.model.population import Population
from evobench.model.solution import Solution
from evobench.util import deshuffle_solution


class Benchmark(ABC):
    """
    Base class for problem encapsulation.
    If you wish to implement your own problem, please
    inherit from this class.
    """

    def __init__(
        self,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Benchmark, self).__init__()
        self.ffe = 0
        self.USE_SHUFFLE = use_shuffle
        self.MULTIPROCESSING = multiprocessing
        self.VERBOSE = verbose

    @abstractproperty
    def genome_size(self) -> int:
        pass

    @lazy
    def gene_order(self) -> np.ndarray:
        gene_order = np.arange(self.genome_size)

        if self.USE_SHUFFLE:
            np.random.shuffle(gene_order)

        return gene_order

    @lazy
    def lower_bound(self) -> np.ndarray:
        pass

    @lazy
    def upper_bound(self) -> np.ndarray:
        pass

    @lazy
    def bound_range(self) -> np.ndarray:
        return self.upper_bound - self.lower_bound

    @lazy
    def as_dict(self) -> Dict:
        """
        Benchmark description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}

        as_dict['name'] = self.__class__.__name__
        as_dict['genome_size'] = self.genome_size
        as_dict['shuffle'] = self.USE_SHUFFLE

        return as_dict

    def random_solution(self) -> Solution:
        pass

    def initialize_population(self, population_size: int) -> Population:
        solutions = []
        population_size = int(population_size)
        iterator = range(population_size)

        if self.VERBOSE:
            tqdm.write('\n')
            iterator = tqdm(iterator, desc='Initializing population')

        for _ in iterator:
            solution = self.random_solution()
            solutions.append(solution)

        return Population(solutions)

    def fix(self, solution: Solution) -> Solution:
        genome = solution.genome.copy()

        mask = genome > self.upper_bound
        genome[mask] = self.upper_bound[mask]

        mask = genome < self.lower_bound
        genome[mask] = self.lower_bound[mask]

        return Solution(genome)

    def predict(self, population: np.ndarray) -> np.ndarray:
        """
        This method is meant to cheat on XAI methods, which require object
        to have `predict` function. This runs just your standard evaluation.

        Parameters
        ----------
        population : np.ndarray
            You can get it using .as_ndarray on your Population object.

        Returns
        -------
        np.ndarray
            Calculated fitness.
        """

        solutions = [Solution(genome) for genome in population]
        population = Population(solutions)
        fitness = self.evaluate_population(population)
        return fitness

    def evaluate_population(self, population: Population) -> np.ndarray:
        """
        Evaluates population of solutions.

        Parameters
        ----------
        population : Population
            Collection of solutions wrapped as `Population`.

        Returns
        -------
        np.ndarray
            An array of fitness values.
            Order is the same as input population.
        """

        solutions = population.get_not_evaluated_solutions()

        if self.VERBOSE:
            tqdm.write('\n')
            tqdm.write(
                'Evaluating population of {} solutions'
                .format(population.size)
            )
            tqdm.write('\n')

            solutions = tqdm(solutions)

        if self.MULTIPROCESSING:
            pool = Pool()
            manager = Manager()
            lock = manager.RLock()

            fitness_map = pool.map(
                partial(
                    self.evaluate_solution,
                    gene_order=self.gene_order,
                    lock=lock
                ),
                solutions
            )

            for solution, fitness in zip(solutions, fitness_map):
                solution.fitness = fitness

        else:
            for solution in solutions:
                solution.fitness = self.evaluate_solution(solution)

        return population.fitness

    def evaluate_solution(
        self,
        solution: Solution,
        gene_order: np.ndarray = None,
        lock: RLock = None,
    ) -> float:
        """
        Evaluate fitness of a single solution.

        Parameters
        ----------
        solution : Solution
            Genome wrapped as `Solution`.
        gene_order: np.ndarray
            When using multiprocessing lazy values aren't copied over the vms.
            This can lead to big whoopsy, when using different
            shuffle order on multiple vms.
        lock : RLock, optional
            Lock to access ffe counter, by default None

        Returns
        -------
        float
            Fitness value.
        """

        if gene_order is None:
            gene_order = self.gene_order

        if lock:
            with lock:
                self.ffe += 1
        else:
            self.ffe += 1

        if self.USE_SHUFFLE:
            solution = deshuffle_solution(solution, gene_order)

        return self._evaluate_solution(solution)

    @abstractmethod
    def _evaluate_solution(self, solution: Solution) -> float:
        pass
