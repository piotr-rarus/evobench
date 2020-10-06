import math
from typing import List

import numpy as np
from lazy import lazy

from evobench.continuous.continuous import Continuous


class Schwefel(Continuous):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Schwefel, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing,
            verbose
        )

    @lazy
    def global_opt(self) -> float:
        # TODO
        pass

    @lazy
    def lower_bound(self) -> np.ndarray:
        lower_bound = [-500] * self.genome_size
        return np.array(lower_bound)

    @lazy
    def upper_bound(self) -> np.ndarray:
        upper_bound = [500] * self.genome_size
        return np.array(upper_bound)

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        fitness = 418.9829 * block.size

        for x in block:
            fitness -= x * math.sin(math.sqrt(abs(x)))

        return fitness
