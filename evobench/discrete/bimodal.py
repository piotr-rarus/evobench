import numpy as np
from evobench.discrete.discrete import Discrete
from evobench.separable import Separable


class Bimodal(Separable, Discrete):

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
