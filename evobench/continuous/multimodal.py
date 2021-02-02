import math
from typing import List

import numpy as np

from evobench.continuous.continuous import Continuous


class Multimodal(Continuous):

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Multimodal, self).__init__(
            blocks,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)
        fitness = abs(math.sin(s))

        return fitness
