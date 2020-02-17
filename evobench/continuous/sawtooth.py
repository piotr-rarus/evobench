import math

import numpy as np

from evobench.separable import Separable


class Sawtooth(Separable):

    def __init__(
        self,
        block_size: int,
        repetitions: int,
        overlap_size: int = 0
    ):
        super(Sawtooth, self).__init__(block_size, repetitions, overlap_size)

        self.GLOBAL_OPTIMUM = 1

    def evaluate_block(self, block: np.ndarray) -> float:
        s = np.sum(block)
        fitness = s - math.floor(s)

        return fitness
