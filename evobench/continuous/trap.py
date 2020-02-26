from typing import List

import numpy as np

from evobench.separable import Separable


class Trap(Separable):

    def __init__(self, blocks: List[int], overlap_size: int = 0):
        super(Trap, self).__init__(blocks, overlap_size)

        self.GLOBAL_OPTIMUM = float('inf')

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)

        fitness = 0

        if s == 0:
            fitness = self.GLOBAL_OPTIMUM
        elif s < 1:
            fitness = 1 / s
        else:
            fitness = s

        return fitness
