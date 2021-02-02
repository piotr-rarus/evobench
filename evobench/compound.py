import itertools
from typing import Dict, List

import numpy as np
from lazy import lazy

from evobench.benchmark import Benchmark
from evobench.continuous.continuous import Continuous
from evobench.discrete.discrete import Discrete
from evobench.model.solution import Solution
from evobench.separable import Separable
from evobench.util import shuffle, shuffle_solution


class CompoundBenchmark(Benchmark):

    def __init__(
        self,
        benchmarks: List[Benchmark],
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(CompoundBenchmark, self).__init__(
            use_shuffle=use_shuffle,
            multiprocessing=multiprocessing,
            verbose=verbose
        )

        self.benchmarks = benchmarks

    @lazy
    def genome_size(self) -> int:
        genome_sizes = [benchmark.genome_size for benchmark in self.benchmarks]
        return sum(genome_sizes)

    @lazy
    def lower_bound(self) -> np.ndarray:
        lower_bounds = [
            list(benchmark.lower_bound)
            for benchmark in self.benchmarks
        ]
        lower_bound = list(itertools.chain(*lower_bounds))
        lower_bound = np.array(lower_bound)

        if self.USE_SHUFFLE:
            lower_bound = shuffle(lower_bound, self.gene_order)

        return lower_bound

    @lazy
    def upper_bound(self) -> np.ndarray:
        upper_bounds = [
            list(benchmark.upper_bound)
            for benchmark in self.benchmarks
        ]
        upper_bound = list(itertools.chain(*upper_bounds))
        upper_bound = np.array(upper_bound)

        if self.USE_SHUFFLE:
            upper_bound = shuffle(upper_bound, self.gene_order)

        return upper_bound

    @lazy
    def as_dict(self) -> Dict:
        """
        Initialization description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}
        as_dict['benchmarks'] = {}
        as_dict['benchmarks']['discrete'] = []
        as_dict['benchmarks']['continuous'] = []

        for benchmark in self.benchmarks:
            if isinstance(benchmark, Discrete):
                as_dict['benchmarks']['discrete'].append(benchmark.as_dict)
            elif isinstance(benchmark, Continuous):
                as_dict['benchmarks']['continuous'].append(benchmark.as_dict)

        benchmark_as_dict = super(CompoundBenchmark, self).as_dict
        as_dict = {**benchmark_as_dict, **as_dict}

        return as_dict

    def random_solution(self) -> Solution:
        genomes = []
        for benchmark in self.benchmarks:
            solution = benchmark.random_solution()
            genomes.append(solution.genome)

        genome = np.concatenate(genomes)
        solution = Solution(genome)

        if self.USE_SHUFFLE:
            solution = shuffle_solution(solution, self.gene_order)

        return solution

    def _evaluate_solution(self, solution: Solution) -> float:
        index = 0
        fitness = 0

        for benchmark in self.benchmarks:
            sub_solution = solution.genome[index:index+benchmark.genome_size]
            sub_solution = Solution(sub_solution)
            fitness += benchmark._evaluate_solution(sub_solution)

            index += benchmark.genome_size

        return fitness

    @lazy
    def true_dsm(self) -> np.ndarray:
        all_separable = all(
            isinstance(benchmark, Separable)
            for benchmark in self.benchmarks
        )

        if not all_separable:
            raise AssertionError(
                'All benchmarks must be separable to get coumpund DSM.'
            )

        start = 0
        dsm = np.zeros((self.genome_size, self.genome_size))
        blocks = [benchmark.BLOCKS for benchmark in self.benchmarks]
        blocks = list(itertools.chain(*blocks))

        overlaps = [
            [benchmark.OVERLAP_SIZE] * len(benchmark.BLOCKS)
            for benchmark in self.benchmarks
        ]
        overlaps = list(itertools.chain(*overlaps))
        iterator = enumerate(zip(blocks, overlaps))

        for index, (block_size, overlap_size) in iterator:

            width = start + block_size
            dsm[start:width, start:width] = 1.0

            start += block_size - index * overlap_size

        return dsm
