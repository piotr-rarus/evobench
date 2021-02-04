import math

import numpy as np
from evobench.continuous.continuous import Continuous
from evobench.separable import Separable


class Multimodal(Separable, Continuous):

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)
        fitness = abs(math.sin(s))

        return fitness
