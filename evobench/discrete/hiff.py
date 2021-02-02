from typing import List

import numpy as np

from evobench.discrete.discrete import Discrete


class Hiff(Discrete):

    def __init__(
        self,
        blocks: List[int],
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Hiff, self).__init__(
            blocks,
            blocks_scaling,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:

        level = 1
        fitness = self.evaluate_level(block, level)

        while block.size > 1:
            block = block.reshape((block.size // 2, 2))

            next_block = []

            for a, b in block:
                next_gene = a * b
                next_block.append(next_gene)

            block = np.array(next_block)
            level *= 2
            fitness += self.evaluate_level(block, level)

        return fitness

    def evaluate_level(self, block: np.array, level: int) -> int:
        s = np.sum(block)
        return level * s
