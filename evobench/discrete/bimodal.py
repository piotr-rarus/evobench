from typing import List

import numpy as np

from evobench.discrete.discrete import Discrete


class Bimodal(Discrete):

    def __init__(
        self,
        blocks: List[int],
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Bimodal, self).__init__(
            blocks,
            blocks_scaling,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        fitness = 0

        if not block.any():
            fitness = block.size // 2
        else:
            half_range = block.size // 2

            unitation = np.count_nonzero(block)

            if unitation == block.size:
                fitness = block.size // 2

            elif unitation < half_range:
                fitness = unitation - 1

            elif unitation >= half_range:
                fitness = block.size - unitation - 1

        return fitness
