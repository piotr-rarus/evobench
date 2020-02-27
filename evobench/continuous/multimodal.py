import math
from typing import List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class Multimodal(Separable):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        super(Multimodal, self).__init__(
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
        fitness = abs(math.sin(s))

        return fitness
