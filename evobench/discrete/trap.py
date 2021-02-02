import numpy as np
from evobench.discrete.discrete import Discrete
from evobench.separable import Separable


class Trap(Separable, Discrete):

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        if not block.any():
            return block.size
        else:
            return np.count_nonzero(block) - 1
