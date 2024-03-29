from typing import Dict, List

import numpy as np
from lazy import lazy

from evobench.discrete.discrete import Discrete
from evobench.separable import Separable


class StepBimodal(Separable, Discrete):

    def __init__(
        self,
        blocks: List[int],
        step_size: int,
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(StepBimodal, self).__init__(
            blocks=blocks,
            blocks_scaling=blocks_scaling,
            overlap_size=overlap_size,
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose
        )

        self.STEP_SIZE = step_size

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        fitness = 0
        unitation = np.count_nonzero(block)
        half_range = block.size // 2

        # global opt
        if not block.any():
            fitness = block.size // 2 // self.STEP_SIZE + 1

        # global opt
        elif unitation == block.size:
            fitness = block.size // 2 // self.STEP_SIZE + 1

        elif unitation < half_range:
            fitness = (unitation + 1) // self.STEP_SIZE - 1

        elif unitation >= half_range:
            unitation = block.size - unitation + 1
            fitness = unitation // self.STEP_SIZE - 1

        return fitness

    @lazy
    def as_dict(self) -> Dict:
        as_dict = {}
        as_dict['step_size'] = self.STEP_SIZE

        separable_as_dict = super().as_dict
        as_dict = {**separable_as_dict, **as_dict}

        return as_dict
