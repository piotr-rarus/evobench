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
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(Continuous, self).__init__(
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
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
        # genomes = self.rng.uniform(
        #     low=0,
        #     high=1,
        #     size=(population_size, self.genome_size)
        # )

        # genomes *= self.bound_range
        # genomes += self.lower_bound

        genomes = self.rng.uniform(
            low=self.lower_bound,
            high=self.upper_bound,
            size=(population_size, self.genome_size)
        )

        if self.USE_SHUFFLE:
            genomes = shuffle(genomes, self.gene_order)

        return list(Solution(genome) for genome in genomes)
