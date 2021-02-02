import numpy as np
from evobench.continuous.continuous import Continuous
from evobench.separable import Separable


class Trap(Separable, Continuous):

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)

        fitness = 0

        if s == 0:
            fitness = float('inf')
        elif s < 1:
            fitness = 1 / s
        else:
            fitness = s

        return fitness
