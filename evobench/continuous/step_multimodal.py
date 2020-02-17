import math

import numpy as np

from evobench.separable import Separable


class StepMultimodal(Separable):

    def __init__(
        self,
        block_size: int,
        repetitions: int,
        step_size: float,
        overlap_size: int = 0
    ):
        super(StepMultimodal, self).__init__(
            block_size,
            repetitions,
            overlap_size
        )

        self.STEP_SIZE = step_size
        self.GLOBAL_OPTIMUM = 1

    def evaluate_block(self, block: np.ndarray) -> float:
        s = np.sum(block)
        fitness = abs(math.sin(s))

        fitness /= self.STEP_SIZE
        fitness = int(fitness)
        fitness *= self.STEP_SIZE

        return fitness
