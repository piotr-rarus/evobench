from typing import List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class StepBimodal(Separable):

    def __init__(
        self,
        blocks: List[int],
        step_size: int,
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        super(StepBimodal, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing
        )

        self.STEP_SIZE = step_size

    @lazy
    def global_opt(self) -> float:

        global_opt = sum(
            block // 2 // self.STEP_SIZE + 1
            for block in self.BLOCKS
        )

        return float(global_opt)

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        # global opt
        if not block.any():
            return block.size // 2 // self.STEP_SIZE + 1

        unitation = np.count_nonzero(block)

        # global opt
        if unitation == block.size:
            return block.size // 2 // self.STEP_SIZE + 1

        half_range = block.size // 2

        if unitation < half_range:
            return (unitation + 1) // self.STEP_SIZE - 1

        if unitation >= half_range:
            unitation = block.size - unitation + 1
            return unitation // self.STEP_SIZE - 1

        return 0
