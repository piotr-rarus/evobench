from typing import List

import numpy as np
from lazy import lazy

from evobench.model.solution import Solution
from evobench.separable import Separable


class Continuous(Separable):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Continuous, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing,
            verbose
        )

    @lazy
    def lower_bound(self) -> np.ndarray:
        lower_bound = [0.0] * self.genome_size
        return np.array(lower_bound)

    @lazy
    def upper_bound(self) -> np.ndarray:
        upper_bound = [1.0] * self.genome_size
        return np.array(upper_bound)

    def random_solution(self) -> Solution:
        genome = np.random.uniform(low=0, high=1, size=self.genome_size)

        genome *= self.bound_range
        genome -= self.lower_bound

        return Solution(genome)
