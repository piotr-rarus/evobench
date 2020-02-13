import numpy as np
from pytest import fixture

from evobench.model import Solution

from ..hiff import Hiff

__BLOCK_SIZE = 8
__REPETITIONS = 1


__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1], 12),
    ([1, 1, 0, 0, 1, 0, 0, 1], 1),
]


@fixture
def hiff() -> Hiff:
    return Hiff(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_samples(hiff: Hiff):
    for genome, score in __SAMPLES:
        solution = Solution(np.array(genome))
        pred_score = hiff.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
