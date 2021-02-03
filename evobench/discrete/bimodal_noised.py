import numpy as np
from evobench.discrete.discrete import Discrete
from evobench.separable import Separable


class BimodalNoised(Separable, Discrete):

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        unitation = np.count_nonzero(block)

        fitness_map = {
            0: 4,
            1: 0,
            2: 2,
            3: 1,
            4: 3,
            5: 2,
            6: 3,
            7: 1,
            8: 2,
            9: 0,
            10: 4
        }

        return fitness_map[unitation]
