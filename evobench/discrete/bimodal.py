from typing import List

import numpy as np

from evobench.discrete.discrete import Discrete


class Bimodal(Discrete):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Bimodal, self).__init__(
            blocks,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

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
