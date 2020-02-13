import numpy as np
from pytest import fixture

from evobench.discrete.bimodal import Bimodal
from evobench.model import Solution

__BLOCK_SIZE = 6
__REPETITIONS = 2

__GLOBAL_OPTIMUM = 6
__LOCAL_OPTIMUM = 4

samples = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 6),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6),
    ([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1], 3)
]


@fixture
def bimodal() -> Bimodal:
    return Bimodal(__BLOCK_SIZE, __REPETITIONS)


def test_samples(bimodal: Bimodal):
    for genome, score in samples:
        solution = Solution(np.array(genome))
        pred_score = bimodal.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
