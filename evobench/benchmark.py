from abc import ABC, abstractmethod, abstractproperty
from multiprocessing import Pool
from typing import Dict

import numpy as np
from lazy import lazy

from evobench.model.population import Population
from evobench.model.solution import Solution


class Benchmark(ABC):

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
        fitness = pool.map(self.evaluate_solution, population.solutions)
        fitness = np.array(fitness, dtype=np.float16)

        return fitness

    @abstractmethod
    def evaluate_solution(self, solution: Solution) -> float:
        pass
