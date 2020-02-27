from typing import List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class Bimodal(Separable):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        super(Bimodal, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing
        )

    @lazy
    def global_opt(self) -> float:
        global_opt = sum(block // 2 for block in self.BLOCKS)
        return float(global_opt)

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:

        if not block.any():
            return block.size // 2

        half_range = block.size // 2

        unitation = np.count_nonzero(block)

        if unitation == block.size:
            return block.size // 2

        if unitation < half_range:
            return unitation - 1

        if unitation >= half_range:
            return block.size - unitation - 1

        return 0
