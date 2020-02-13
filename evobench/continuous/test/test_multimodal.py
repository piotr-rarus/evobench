import numpy as np
from pytest import fixture

from evobench.model import Solution

from ..multimodal import Multimodal

__BLOCK_SIZE = 5
__REPETITIONS = 2


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def multimodal() -> Multimodal:
    return Multimodal(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_samples(multimodal: Multimodal):
    for genome, score in __SAMPLES:
        solution = Solution(np.array(genome))
        pred_score = multimodal.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
