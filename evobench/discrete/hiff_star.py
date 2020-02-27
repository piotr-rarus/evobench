from typing import List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class HiffStar(Separable):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False
    ):
        super(HiffStar, self).__init__(blocks, overlap_size, shuffle)

    @lazy
    def global_opt(self) -> float:

        global_opt = sum(
            block // 2 * (block // 2 - 1)
            for block in self.BLOCKS
        )

        return float(global_opt)

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
