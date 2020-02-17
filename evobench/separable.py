from abc import abstractmethod
from typing import Dict

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
        block_size: int,
        repetitions: int,
        overlap_size: int = 0
    ):
        """
        Parameters
        ----------
        block_size : int
            Size of a single block.
        repetitions : int
            Number of concats.
            That many times your base problem will be repeated.
        overlap_size : int, optional
            That many genes will overlap between different blocks, by default 0
        """

        super(Separable, self).__init__()

        self.BLOCK_SIZE = block_size
        self.REPETITIONS = repetitions
        self.OVERLAP_SIZE = overlap_size

        self.GENOME_SIZE = block_size * repetitions
        self.GENOME_SIZE -= (repetitions - 1) * self.OVERLAP_SIZE

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
        as_dict['block_size'] = self.BLOCK_SIZE
        as_dict['repetitions'] = self.REPETITIONS
        as_dict['overlap_size'] = self.OVERLAP_SIZE

        benchmark_as_dict = super().as_dict
        as_dict = {**benchmark_as_dict, **as_dict}

        return as_dict

    def _evaluate_solution(self, solution: Solution) -> float:

        blocks = []

        for r in range(self.REPETITIONS):
            start = r * self.BLOCK_SIZE - r * self.OVERLAP_SIZE
            block = solution.genome[start:start+self.BLOCK_SIZE]
            blocks.append(block)

        score = sum(self.evaluate_block(block) for block in blocks)

        return float(score)

    @abstractmethod
    def evaluate_block(self, block: np.ndarray) -> float:
        """
        Base evaluation of single block.
        If you wish to implement your own problem, please implement this.

        Parameters
        ----------
        block : np.ndarray
            Separated genome slice of your problem.

        Returns
        -------
        float
            Fitness value of a block.
        """
        pass
