from typing import List, Tuple

import numpy as np

from evobench.benchmark import Benchmark
from evobench.model.solution import Solution


def check_samples(
    samples: List[Tuple[np.ndarray, float]],
    benchmark: Benchmark
):
    for genome, fitness in samples:
        solution = Solution(np.array(genome))
        pred_fitness = benchmark.evaluate_solution(solution)

        assert isinstance(pred_fitness, float)
        assert pred_fitness == fitness
