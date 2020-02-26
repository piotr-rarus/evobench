from typing import List

import numpy as np

from evobench.separable import Separable


class StepBimodal(Separable):

    def __init__(
        self,
        blocks: List[int],
        step_size: int,
        overlap_size: int = 0
    ):
        super(StepBimodal, self).__init__(blocks, overlap_size)

        self.STEP_SIZE = step_size

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        if not block.any():
            return block.size // 2 // self.STEP_SIZE + 1  # global opt

        unitation = np.count_nonzero(block)

        if unitation == self.BLOCK_SIZE:
            return block.size // 2 // self.STEP_SIZE + 1  # global opt

        half_range = block.size // 2

        if unitation < half_range:
            return (unitation + 1) // self.STEP_SIZE - 1

        if unitation >= half_range:
            unitation = block.size - unitation + 1
            return unitation // self.STEP_SIZE - 1

        return 0
