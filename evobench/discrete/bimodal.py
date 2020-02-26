from typing import List

import numpy as np

from evobench.separable import Separable


class Bimodal(Separable):

    def __init__(self, blocks: List[int], overlap_size: int = 0):

        super(Bimodal, self).__init__(blocks, overlap_size)
        self.GLOBAL_OPTIMUM = sum(block.size // 2 for block in blocks)

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:

        if not block.any():
            return block.size // 2

        half_range = block.size // 2

        unitation = np.count_nonzero(block)

        if unitation == block.size:
            return block.size

        if unitation < half_range:
            return unitation - 1

        if unitation >= half_range:
            return block.size - unitation - 1

        return 0
