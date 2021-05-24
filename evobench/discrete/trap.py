import numpy as np

from evobench.discrete.discrete import Discrete
from evobench.separable import Separable


class Trap(Separable, Discrete):

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        non_zero_count = np.count_nonzero(block)
        if not non_zero_count:
            return block.size
        else:
            return non_zero_count - 1
