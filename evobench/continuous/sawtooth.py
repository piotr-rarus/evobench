import math
from typing import List

import numpy as np

from evobench.separable import Separable


class Sawtooth(Separable):

    def __init__(self, blocks: List[int], overlap_size: int = 0):
        super(Sawtooth, self).__init__(blocks, overlap_size)

        self.GLOBAL_OPTIMUM = 1

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)
        fitness = s - math.floor(s)

        return fitness
