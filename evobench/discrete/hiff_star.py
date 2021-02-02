import numpy as np
from evobench.discrete.discrete import Discrete
from evobench.separable import Separable


class HiffStar(Separable, Discrete):

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
