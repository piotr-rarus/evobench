from typing import Dict, List

import numpy as np
from lazy import lazy

from evobench.continuous.continuous import Continuous
from evobench.separable import Separable


class StepTrap(Separable, Continuous):

    def __init__(
        self,
        *,
        blocks: List[int],
        step_size: int,
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(StepTrap, self).__init__(
            blocks=blocks,
            blocks_scaling=blocks_scaling,
            overlap_size=overlap_size,
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose
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
