import numpy as np
from evobench.discrete.discrete import Discrete
from evobench.separable import Separable


class Hiff(Separable, Discrete):

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
