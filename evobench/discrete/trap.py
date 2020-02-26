from typing import List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class Trap(Separable):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False
    ):
        super(Trap, self).__init__(blocks, overlap_size, shuffle)

    @lazy
    def global_opt(self) -> float:
        return float(sum(self.BLOCKS))

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        if not block.any():
            return block.size
        else:
            return np.count_nonzero(block) - 1
