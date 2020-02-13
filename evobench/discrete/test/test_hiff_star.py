import numpy as np
from pytest import fixture

from evobench.model import Solution

from ..hiff_star import HiffStar

__BLOCK_SIZE = 8
__REPETITIONS = 3
__GLOBAL_OPTIMUM = 36

__SAMPLES = [
    ([1] * __BLOCK_SIZE * __REPETITIONS, __GLOBAL_OPTIMUM),
    ([0] * __BLOCK_SIZE * __REPETITIONS, __GLOBAL_OPTIMUM)
]


@fixture
def hiff_star() -> HiffStar:
    return HiffStar(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_scores(hiff_star: HiffStar):
    for genome, score in __SAMPLES:
        solution = Solution(np.array(genome))
        pred_score = hiff_star.evaluate_solution(solution)

        assert pred_score == __GLOBAL_OPTIMUM
