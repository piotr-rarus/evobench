import numpy as np

from evobench.separable import Separable


class Trap(Separable):

    def __init__(
        self,
        block_size: int,
        repetitions: int,
        overlap_size: int = 0
    ):
        super(Trap, self).__init__(block_size, repetitions, overlap_size)

        self.GLOBAL_OPTIMUM = float('inf')

    def evaluate_block(self, block: np.ndarray) -> float:
        s = np.sum(block)

        fitness = 0

        if s == 0:
            fitness = self.GLOBAL_OPTIMUM
        elif s < 1:
            fitness = 1 / s
        else:
            fitness = s

        return fitness
