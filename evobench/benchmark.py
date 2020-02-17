from abc import ABC, abstractmethod, abstractproperty
from multiprocessing import Pool
from typing import Dict

import numpy as np
from lazy import lazy

from evobench.model.population import Population
from evobench.model.solution import Solution
from multiprocessing import RLock, Manager
from functools import partial


class Benchmark(ABC):

    def __init__(self):
        super(Benchmark, self).__init__()
        self.ffe = 0

    @abstractproperty
    def genome_size(self) -> int:
        pass

    @lazy
    def as_dict(self) -> Dict:
        as_dict = {}

        as_dict['name'] = self.__class__.__name__
        as_dict['genome_size'] = self.genome_size

        return as_dict

    def evaluate_population(self, population: Population) -> np.ndarray:
        pool = Pool()
        manager = Manager()
        lock = manager.RLock()

        fitness = pool.map(
            partial(self.evaluate_solution, lock=lock),
            population.solutions
        )

        fitness = np.array(fitness, dtype=np.float16)

        return fitness

    def evaluate_solution(
        self,
        solution: Solution,
        lock: RLock = None
    ) -> float:
        if lock:
            with lock:
                self.ffe += 1

        return self._evaluate_solution(solution)

    @abstractmethod
    def _evaluate_solution(self, solution: Solution) -> float:
        pass
