import math
from typing import List

import numpy as np
from lazy import lazy
from evobench.separable import Separable


class StepMultimodal(Separable):

    def __init__(
        self,
        blocks: List[int],
        step_size: int,
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        super(StepMultimodal, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing
        )

        self.STEP_SIZE = step_size

    @lazy
    def global_opt(self) -> float:
        return float(len(self.BLOCKS))

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)
        fitness = abs(math.sin(s))

        fitness /= self.STEP_SIZE
        fitness = int(fitness)
        fitness *= self.STEP_SIZE

        return fitness
