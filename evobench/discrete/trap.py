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

        self.GLOBAL_OPTIMUM = self.BLOCK_SIZE

    def evaluate_block(self, block: np.ndarray) -> int:
        if not block.any():
            return self.GLOBAL_OPTIMUM
        else:
            return np.count_nonzero(block) - 1
