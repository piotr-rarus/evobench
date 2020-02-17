from typing import List, Tuple

import numpy as np

from evobench.benchmark import Benchmark
from evobench.model.solution import Solution


def check_samples(
    samples: List[Tuple[np.ndarray, float]],
    benchmark: Benchmark
):
    for genome, score in samples:
        solution = Solution(np.array(genome))
        pred_score = benchmark.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
