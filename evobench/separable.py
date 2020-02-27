from abc import abstractmethod
from typing import Dict, List

import numpy as np
from lazy import lazy

from evobench.benchmark import Benchmark
from evobench.model import Solution


class Separable(Benchmark):

    """
    Base class for fully separable problems.
    """

    def __init__(
        self,
        blocks: List[int],
        overlap_size: int = 0,
        shuffle: bool = False,
        multiprocessing: bool = False
    ):
        """
        Parameters
        ----------
        blocks : int
            Sizes of each block
        overlap_size : int, optional
            That many genes will overlap between different blocks, by default 0
        shuffle : bool, optional
            Whether to shuffle the genome, by default False
        multiprocessing : bool, optional
            Whether to evaluate population on all cores, by default False
        """

        super(Separable, self).__init__(shuffle, multiprocessing)

        self.BLOCKS = blocks
        self.OVERLAP_SIZE = overlap_size

        self.GENOME_SIZE = sum(blocks)
        self.GENOME_SIZE -= (len(blocks) - 1) * self.OVERLAP_SIZE

    @lazy
    def genome_size(self) -> int:
        return self.GENOME_SIZE

    @lazy
    def as_dict(self) -> Dict:
        """
        Initialization description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}
        as_dict['blocks'] = self.BLOCKS
        as_dict['overlap_size'] = self.OVERLAP_SIZE

        benchmark_as_dict = super().as_dict
        as_dict = {**benchmark_as_dict, **as_dict}

        return as_dict

    def _evaluate_solution(self, solution: Solution) -> float:

        blocks = []
        start = 0

        for index, block_size in enumerate(self.BLOCKS):

            block = solution.genome[start: start + block_size]
            blocks.append(block)

            start += block_size - index * self.OVERLAP_SIZE

        score = sum(
            self.evaluate_block(block, index)
            for index, block
            in enumerate(blocks)
        )

        return float(score)

    @abstractmethod
    def evaluate_block(self, block: np.ndarray, block_index: int) -> float:
        """
        Base evaluation of single block.
        If you wish to implement your own problem, please implement this.

        Parameters
        ----------
        block : np.ndarray
            Separated genome slice of your problem.
        block_index : int

        Returns
        -------
        float
            Fitness value of a block.
        """
        pass
