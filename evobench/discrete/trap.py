from typing import List

import numpy as np

from evobench.discrete.discrete import Discrete


class Trap(Discrete):

    def __init__(
        self,
        blocks: List[int],
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Trap, self).__init__(
            blocks,
            blocks_scaling,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        if not block.any():
            return block.size
        else:
            return np.count_nonzero(block) - 1
