from typing import List

import numpy as np
from lazy import lazy

from evobench.discrete.discrete import Discrete


class Trap(Discrete):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Trap, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing,
            verbose
        )

    @lazy
    def global_opt(self) -> float:
        return float(sum(self.BLOCKS))

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        if not block.any():
            return block.size
        else:
            return np.count_nonzero(block) - 1
