import numpy as np

from evobench.separable import Separable


class StepBimodal(Separable):

    def __init__(
        self,
        block_size: int,
        repetitions: int,
        step_size: int,
        overlap_size: int = 0
    ):
        super(StepBimodal, self).__init__(
            block_size,
            repetitions,
            overlap_size
        )

        self.STEP_SIZE = step_size

        self.LOCAL_OPTIMUM = self.BLOCK_SIZE // 2 // step_size
        self.GLOBAL_OPTIMUM = self.LOCAL_OPTIMUM + 1
        self.HALF_RANGE = self.BLOCK_SIZE // 2

    def evaluate_block(self, block: np.ndarray) -> int:
        if not block.any():
            return self.GLOBAL_OPTIMUM

        unitation = np.count_nonzero(block)

        if unitation == self.BLOCK_SIZE:
            return self.GLOBAL_OPTIMUM

        if unitation < self.HALF_RANGE:
            return (unitation + 1) // self.STEP_SIZE - 1

        if unitation >= self.HALF_RANGE:
            unitation = self.BLOCK_SIZE - unitation
            return (unitation + 1) // self.STEP_SIZE - 1

        return 0
