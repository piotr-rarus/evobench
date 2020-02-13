import numpy as np

from evobench.separable import Separable


class HiffStar(Separable):

    def __init__(
        self,
        block_size: int,
        repetitions: int,
        overlap_size: int = 0
    ):
        super(HiffStar, self).__init__(block_size, repetitions, overlap_size)

    def evaluate_block(self, block: np.ndarray) -> int:

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
