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

    def __init__(self, shuffle: bool = False):
        super(Benchmark, self).__init__()
        self.ffe = 0
        self.shuffle = shuffle

    @abstractproperty
    def genome_size(self) -> int:
        pass

    @abstractproperty
    def global_opt(self) -> float:
        pass

    @lazy
    def gene_order(self) -> List[int]:
        gene_order = range(0, self.genome_size)

        if self.shuffle:
            gene_order = np.random.permutation(gene_order)

        return gene_order

    @lazy
    def as_dict(self) -> Dict:
        """
        Benchmark description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}

        as_dict['name'] = self.__class__.__name__
        as_dict['genome_size'] = self.genome_size
        as_dict['shuffle'] = self.shuffle

        return as_dict

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

        pool = Pool()
        manager = Manager()
        lock = manager.RLock()

        tqdm.write('\n')
        tqdm.write(
            'Evaluating population of {} solutions'
            .format(population.length)
        )
        tqdm.write('\n')

        fitness = pool.map(
            partial(self.evaluate_solution, lock=lock),
            tqdm(population.solutions)
        )

        fitness = np.array(fitness, dtype=np.float16)

        return fitness

    def evaluate_solution(
        self,
        solution: Solution,
        lock: RLock = None
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

        if lock:
            with lock:
                self.ffe += 1
        else:
            self.ffe += 1

        if self.shuffle:
            shuffled = []

            for gene_index in self.gene_order:
                shuffled.append(solution.genome[gene_index])

            solution = Solution(np.array(shuffled))

        return self._evaluate_solution(solution)

    @abstractmethod
    def _evaluate_solution(self, solution: Solution) -> float:
        pass
