from typing import List

import numpy as np
from lazy import lazy

from evobench.benchmark import Benchmark
from evobench.model.solution import Solution
from evobench.util import shuffle


class Continuous(Benchmark):

    def __init__(
        self,
        *,
        random_state: int = 42,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        super(Continuous, self).__init__(
            random_state=random_state,
            use_shuffle=use_shuffle,
            multiprocessing=multiprocessing,
            verbose=verbose
        )

    @lazy
    def lower_bound(self) -> np.ndarray:
        lower_bound = [0.0] * self.genome_size
        return np.array(lower_bound)

    @lazy
    def upper_bound(self) -> np.ndarray:
        upper_bound = [1.0] * self.genome_size
        return np.array(upper_bound)

    def random_solutions(self, population_size: int) -> List[Solution]:
        genomes = self.random_state.uniform(
            low=0,
            high=1,
            size=(population_size, self.genome_size)
        )

        genomes *= self.bound_range
        genomes += self.lower_bound

        if self.USE_SHUFFLE:
            genomes = shuffle(genomes, self.gene_order)

        return list(Solution(genome) for genome in genomes)
