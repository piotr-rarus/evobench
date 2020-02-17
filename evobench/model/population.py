from dataclasses import dataclass
from typing import List

import numpy as np
from lazy import lazy

from .solution import Solution


@dataclass(frozen=True)
class Population:
    solutions: List[Solution]

    @lazy
    def as_ndarray(self) -> np.ndarray:
        population = [solution.genome for solution in self.solutions]
        population = np.array(population, dtype=np.float16)

        return population
