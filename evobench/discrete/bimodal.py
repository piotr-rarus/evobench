import numpy as np

from evobench.separable import Separable


class Bimodal(Separable):

    def __init__(
        self,
        block_size: int,
        repetitions: int,
        overlap_size: int = 0
    ):
        super(Bimodal, self).__init__(block_size, repetitions, overlap_size)

        self.GLOBAL_OPTIMUM = self.BLOCK_SIZE // 2
        self.LOCAL_OPTIMUM = self.GLOBAL_OPTIMUM - 1
        self.HALF_RANGE = self.BLOCK_SIZE // 2

    def evaluate_block(self, block: np.ndarray) -> int:
        if not block.any():
            return self.GLOBAL_OPTIMUM

        unitation = np.count_nonzero(block)

        if unitation == self.BLOCK_SIZE:
            return self.GLOBAL_OPTIMUM

        if unitation < self.HALF_RANGE:
            return unitation - 1

        if unitation >= self.HALF_RANGE:
            unitation = self.BLOCK_SIZE - unitation
            return unitation - 1

        return 0
