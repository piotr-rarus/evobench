from typing import Dict, List

import numpy as np
from lazy import lazy

from evobench.separable import Separable


class StepTrap(Separable):

    def __init__(
        self,
        blocks: List[int],
        step_size: int,
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        super(StepTrap, self).__init__(
            blocks,
            overlap_size,
            shuffle,
            multiprocessing
        )

        self.STEP_SIZE = step_size

    @lazy
    def global_opt(self) -> float:
        global_opt = sum(block // self.STEP_SIZE for block in self.BLOCKS)
        return float(global_opt)

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        if not block.any():
            return block.size // self.STEP_SIZE
        else:
            return (np.count_nonzero(block) + 1) // self.STEP_SIZE - 1

    @lazy
    def as_dict(self) -> Dict:
        as_dict = {}
        as_dict['step_size'] = self.STEP_SIZE

        separable_as_dict = super().as_dict
        as_dict = {**separable_as_dict, **as_dict}

        return as_dict
