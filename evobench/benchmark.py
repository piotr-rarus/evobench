from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from multiprocessing import Manager, Pool, RLock
from typing import Dict

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

    def __init__(self):
        super(Benchmark, self).__init__()
        self.ffe = 0

    @abstractproperty
    def genome_size(self) -> int:
        pass

    @lazy
    def as_dict(self) -> Dict:
        """
        Benchmark description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}

        as_dict['name'] = self.__class__.__name__
        as_dict['genome_size'] = self.genome_size

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

        return self._evaluate_solution(solution)

    @abstractmethod
    def _evaluate_solution(self, solution: Solution) -> float:
        pass
