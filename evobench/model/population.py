from dataclasses import dataclass
from typing import List

import numpy as np

from .solution import Solution


@dataclass(frozen=True)
class Population:
    solutions: List[Solution]

    @property
    def length(self) -> int:
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
        population = np.array(population)

        return population
