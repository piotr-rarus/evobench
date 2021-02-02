from typing import List

import numpy as np

from evobench.discrete.discrete import Discrete


class HiffStar(Discrete):

    def __init__(
        self,
        blocks: List[int],
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(HiffStar, self).__init__(
            blocks,
            blocks_scaling,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:

        fitness = 0
        level = 1

        while block.size > 1:
            block = block.reshape((block.size // 2, 2))

            next_block = []

            for a, b in block:
                next_gene = -1

                if a == 1 and b == 1:
                    next_gene = 1

                elif a == 0 and b == 0:
                    next_gene = 0

                if next_gene != -1:
                    fitness += level

                next_block.append(next_gene)

            block = np.array(next_block)
            level *= 2

        return fitness
