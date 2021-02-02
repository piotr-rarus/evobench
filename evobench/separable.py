from abc import abstractmethod
from typing import Dict, List

import numpy as np
from lazy import lazy

from evobench.benchmark import Benchmark
from evobench.model import Solution


class Separable(Benchmark):

    """
    Base class for separable problems.
    """

    def __init__(
        self,
        blocks: List[int],
        blocks_scaling: List[int] = None,
        overlap_size: int = 0,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        """
        Parameters
        ----------
        blocks : List[int]
            Sizes of each block
        blocks_scaling : List[int], optional
            Fitness scale factors for each block, by default None
        overlap_size : int, optional
            That many genes will overlap between different blocks, by default 0
        use_shuffle : bool, optional
            Whether to shuffle the genome, by default False
        multiprocessing : bool, optional
            Whether to evaluate population on all cores, by default False
        """

        super(Separable, self).__init__(use_shuffle, multiprocessing, verbose)

        self.BLOCKS = blocks
        self.BLOCKS_SCALING = blocks_scaling
        self.OVERLAP_SIZE = overlap_size

    @lazy
    def genome_size(self) -> int:
        genome_size = sum(self.BLOCKS)
        genome_size -= (len(self.BLOCKS) - 1) * self.OVERLAP_SIZE
        return genome_size

    @lazy
    def as_dict(self) -> Dict:
        """
        Initialization description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}
        as_dict['blocks'] = self.BLOCKS
        as_dict['overlap_size'] = self.OVERLAP_SIZE
        as_dict['block_scaling'] = self.BLOCKS_SCALING

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

        evaluations = [
            self.evaluate_block(block, index)
            for index, block in enumerate(blocks)
        ]

        if self.BLOCKS_SCALING:
            evaluations = [
                evaluation * self.BLOCKS_SCALING[index]
                for index, evaluation in enumerate(evaluations)
            ]

        fitness = sum(evaluations)
        return float(fitness)

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

    @lazy
    def true_dsm(self) -> np.ndarray:
        start = 0
        dsm = np.zeros((self.genome_size, self.genome_size))

        for index, block_size in enumerate(self.BLOCKS):

            width = start + block_size
            dsm[start:width, start:width] = 1.0

            start += block_size - index * self.OVERLAP_SIZE

        return dsm
