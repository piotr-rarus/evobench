import numpy as np
from evobench.model.solution import Solution
from evobench.benchmark import Benchmark
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

    def random_solution(self) -> Solution:
        genome = np.random.uniform(low=0, high=2, size=self.genome_size)

        genome *= self.bound_range
        genome -= self.lower_bound
        genome = genome.astype(dtype=np.int)

        if self.USE_SHUFFLE:
            genome = shuffle(genome, self.gene_order)

        return Solution(genome)
