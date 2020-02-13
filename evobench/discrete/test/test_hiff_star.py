import numpy as np
from pytest import fixture
from typing import Iterator, Tuple, List

from evobench.model import Solution

from ..hiff_star import HiffStar

__BLOCK_SIZE = 8
__REPETITIONS = 3
__GLOBAL_OPTIMUM = 36


@fixture
def samples() -> List[Tuple[np.ndarray, float]]:
    return [
        ([1] * __BLOCK_SIZE * __REPETITIONS, __GLOBAL_OPTIMUM),
        ([0] * __BLOCK_SIZE * __REPETITIONS, __GLOBAL_OPTIMUM)
    ]


def test_scores(samples: Iterator[Tuple[np.ndarray, float]]):

    hiff = HiffStar(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)

    for genome, score in samples:
        genome = np.array(genome)
        solution = Solution(genome)
        pred_score = hiff.evaluate_solution(solution)

        assert pred_score == __GLOBAL_OPTIMUM
