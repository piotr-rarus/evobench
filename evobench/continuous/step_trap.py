from typing import Dict, List

import numpy as np
from evobench.continuous.continuous import Continuous
from lazy import lazy


class StepTrap(Continuous):

    def __init__(
        self,
        blocks: List[int],
        step_size: int,
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(StepTrap, self).__init__(
            blocks,
            blocks_scaling,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

        self.STEP_SIZE = step_size

    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        s = np.sum(block)

        fitness = 0

        if s == 0:
            fitness = float('inf')
        elif s < 1:
            fitness = 1 / s
            fitness /= self.STEP_SIZE
            fitness = int(fitness)
            fitness *= self.STEP_SIZE
        else:
            fitness = s
            fitness /= self.STEP_SIZE
            fitness = int(fitness)
            fitness *= self.STEP_SIZE

        return fitness

    @lazy
    def as_dict(self) -> Dict:
        as_dict = {}
        as_dict['step_size'] = self.STEP_SIZE

        separable_as_dict = super().as_dict
        as_dict = {**separable_as_dict, **as_dict}

        return as_dict
