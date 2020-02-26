from typing import List

import numpy as np

from evobench.separable import Separable


class Hiff(Separable):

    def __init__(self, blocks: List[int], overlap_size: int = 0):
        super(Hiff, self).__init__(blocks, overlap_size)

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:

        fitness = 0
        level = 1

        while block.size > 1:
            block = block.reshape((block.size // 2, 2))

            next_block = []

            for a, b in block:
                next_gene = a * b
                next_block.append(next_gene)

            block = np.array(next_block)
            fitness += self.evaluate_level(block, level)
            level *= 2

        return fitness

    def evaluate_level(self, block: np.array, level: int) -> int:
        s = np.sum(block)
        return level * s
