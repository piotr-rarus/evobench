from typing import List

import numpy as np
from evobench.benchmark import Benchmark
from evobench.model.solution import Solution
from evobench.util import shuffle
from lazy import lazy


class Discrete(Benchmark):

    def __init__(
        self,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Discrete, self).__init__(
            use_shuffle,
            multiprocessing,
            verbose
        )

    @lazy
    def lower_bound(self) -> np.ndarray:
        lower_bound = [0] * self.genome_size
        return np.array(lower_bound)

    @lazy
    def upper_bound(self) -> np.ndarray:
        upper_bound = [1] * self.genome_size
        return np.array(upper_bound)

    def fix(self, solution: Solution) -> Solution:
        solution = super(Discrete, self).fix(solution)
        genome = solution.genome.astype(dtype=np.int)

        return Solution(genome)

    def random_solutions(self, population_size: int) -> List[Solution]:
        genomes = np.random.uniform(
            low=0,
            high=2,
            size=(population_size, self.genome_size)
        )

        genomes *= self.bound_range
        genomes -= self.lower_bound
        genomes = genomes.astype(dtype=np.int)

        if self.USE_SHUFFLE:
            genomes = shuffle(genomes, self.gene_order)

        return list(Solution(genome) for genome in genomes)
