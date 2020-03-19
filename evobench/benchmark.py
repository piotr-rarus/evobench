from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from multiprocessing import Manager, Pool, RLock
from typing import Dict, List

import numpy as np
from lazy import lazy
from tqdm import tqdm

from evobench.model.population import Population
from evobench.model.solution import Solution


class Benchmark(ABC):
    """
    Base class for problem encapsulation.
    If you wish to implement your own problem, please
    inherit from this class.
    """

    def __init__(self, shuffle: bool = False, multiprocessing: bool = False):
        super(Benchmark, self).__init__()
        self.ffe = 0
        self.SHUFFLE = shuffle
        self.MULTIPROCESSING = multiprocessing

    @abstractproperty
    def genome_size(self) -> int:
        pass

    @abstractproperty
    def global_opt(self) -> float:
        pass

    @lazy
    def gene_order(self) -> List[int]:
        gene_order = range(0, self.genome_size)

        if self.SHUFFLE:
            gene_order = np.random.permutation(gene_order)

        return list(gene_order)

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
        as_dict['shuffle'] = self.SHUFFLE

        return as_dict

    @lazy
    def random_solution(self) -> Solution:
        pass

    def initialize_population(self, size: int) -> Population:
        size = int(size)
        solutions = []

        tqdm.write('\n')
        for _ in tqdm(range(size), desc='Initializing population'):
            genome = self.random_solution().genome

            solution = Solution(genome)
            solutions.append(solution)

        return Population(solutions)

    def fix(self, solution: Solution) -> Solution:
        genome = solution.genome.copy()

        mask = genome > self.upper_bound
        genome[mask] = self.upper_bound[mask]

        mask = genome < self.lower_bound
        genome[mask] = self.lower_bound[mask]

        return Solution(genome)

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

        tqdm.write('\n')
        tqdm.write(
            'Evaluating population of {} solutions'
            .format(population.length)
        )
        tqdm.write('\n')

        fitness = None

        if self.MULTIPROCESSING:
            pool = Pool()
            manager = Manager()
            lock = manager.RLock()

            fitness = pool.map(
                partial(
                    self.evaluate_solution,
                    gene_order=self.gene_order,
                    lock=lock
                ),
                tqdm(population.solutions)
            )

        else:
            fitness = [
                self.evaluate_solution(solution)
                for solution in tqdm(population.solutions)
            ]

        return np.array(fitness)

    def evaluate_solution(
        self,
        solution: Solution,
        gene_order: List[int] = None,
        lock: RLock = None,
    ) -> float:
        """
        Evaluate fitness of a single solution.

        Parameters
        ----------
        solution : Solution
            Genome wrapped as `Solution`.
        lock : RLock, optional
            Lock to access ffe counter, by default None

        Returns
        -------
        float
            Fitness value.
        """

        if not gene_order:
            gene_order = self.gene_order

        if lock:
            with lock:
                self.ffe += 1
        else:
            self.ffe += 1

        if self.SHUFFLE:
            solution = self._shuffle_solution(solution, gene_order)

        return self._evaluate_solution(solution)

    def _shuffle_solution(self, solution: Solution, gene_order: List[int]):
        shuffled = []

        for gene_index in gene_order:
            shuffled.append(solution.genome[gene_index])

        return Solution(np.array(shuffled))

    @abstractmethod
    def _evaluate_solution(self, solution: Solution) -> float:
        pass
