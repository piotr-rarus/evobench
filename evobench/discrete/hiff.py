from typing import List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class Hiff(Separable):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        super(Hiff, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing
        )

    @lazy
    def global_opt(self) -> float:

        global_opt = sum(block * block // 2 for block in self.BLOCKS)
        return float(global_opt)

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
