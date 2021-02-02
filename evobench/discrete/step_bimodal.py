from typing import Dict, List

import numpy as np
from evobench.discrete.discrete import Discrete
from lazy import lazy


class StepBimodal(Discrete):

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
        super(StepBimodal, self).__init__(
            blocks,
            blocks_scaling,
            overlap_size,
            use_shuffle,
            multiprocessing,
            verbose
        )

        self.STEP_SIZE = step_size

    def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
        # global opt
        if not block.any():
            return block.size // 2 // self.STEP_SIZE + 1

        unitation = np.count_nonzero(block)

        # global opt
        if unitation == block.size:
            return block.size // 2 // self.STEP_SIZE + 1

        half_range = block.size // 2

        if unitation < half_range:
            return (unitation + 1) // self.STEP_SIZE - 1

        if unitation >= half_range:
            unitation = block.size - unitation + 1
            return unitation // self.STEP_SIZE - 1

        return 0

    @lazy
    def as_dict(self) -> Dict:
        as_dict = {}
        as_dict['step_size'] = self.STEP_SIZE

        separable_as_dict = super().as_dict
        as_dict = {**separable_as_dict, **as_dict}

        return as_dict
