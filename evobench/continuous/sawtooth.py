import math
from typing import List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class Sawtooth(Separable):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        super(Sawtooth, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing
        )

    @lazy
    def global_opt(self) -> float:
        return float(len(self.BLOCKS))

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)
        fitness = s - math.floor(s)

        return fitness
