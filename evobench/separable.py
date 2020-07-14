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
        multiprocessing: bool = False,
        verbose: int = 0
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

        super(Separable, self).__init__(shuffle, multiprocessing, verbose)

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

        fitness = sum(
            self.evaluate_block(block, index)
            for index, block
            in enumerate(blocks)
        )

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
            dsm[start:width, start:width] = 1

            start += block_size - index * self.OVERLAP_SIZE

        return dsm

    def dsm_fill_quality(self, pred_dsm: np.ndarray) -> List[float]:
        # TODO: link the paper

        fill_quality = []

        for gene_index in range(self.genome_size):
            pred_ils = self._get_ils(gene_index, pred_dsm)
            true_ils = self._get_ils(gene_index, self.true_dsm)

            block_width = self._get_block_width(gene_index)

            pred_ils = pred_ils[1:block_width]
            true_ils = true_ils[1:block_width]

            pred_ils = set(pred_ils)
            true_ils = set(true_ils)

            pred_positive = true_ils.intersection(pred_ils)

            quality = len(pred_positive) / len(true_ils)

            fill_quality.append(quality)

        return fill_quality

    def _get_block_width(self, gene_index: int) -> int:
        start = 0

        for index, block_size in enumerate(self. BLOCKS):
            end = start + block_size

            if gene_index < end:
                return block_size

            start += block_size - index * self.OVERLAP_SIZE


    def _get_ils(self, gene_index: int, dsm: np.ndarray):
        # TODO: link the paper
        ils = []
        ils.append(gene_index)

        while len(ils) < self.genome_size:
            last_gene_index = ils[-1]

            available_genes = [
                gene_index
                for gene_index in range(0, self.genome_size)
                if gene_index not in ils
            ]

            dependencies = [
                dsm[last_gene_index, available_gene]
                for available_gene in available_genes
            ]

            max_index = np.argmax(dependencies)
            ils.append(available_genes[max_index])

        return ils
