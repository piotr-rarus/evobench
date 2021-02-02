from dataclasses import dataclass
from typing import List

import numpy as np

from .solution import Solution


@dataclass(frozen=True)
class Population:
    solutions: List[Solution]

    @property
    def size(self) -> int:
        return len(self.solutions)

    @property
    def upper_bound(self) -> np.ndarray:
        return self.as_ndarray.max(axis=0)

    @property
    def lower_bound(self) -> np.ndarray:
        return self.as_ndarray.min(axis=0)

    @property
    def as_ndarray(self) -> np.ndarray:
        population = [solution.genome for solution in self.solutions]
        return np.array(population)

    @property
    def are_all_evaluated(self) -> bool:
        return all(
            solution.fitness is not None
            for solution in self.solutions
        )

    def get_not_evaluated_solutions(self) -> List[Solution]:
        solutions = [
            solution
            for solution in self.solutions
            if solution.fitness is None
        ]

        return list(solutions)

    @property
    def fitness(self) -> np.ndarray:
        if not self.are_all_evaluated:
            raise Exception('Please evaluate your population first.')

        fitness = [solution.fitness for solution in self.solutions]
        return np.array(fitness)
