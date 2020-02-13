import numpy as np
from pytest import fixture

from evobench.model import Solution

from ..sawtooth import Sawtooth

__BLOCK_SIZE = 5
__REPETITIONS = 2

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0),
    ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1),
]


@fixture
def sawtooth() -> Sawtooth:
    return Sawtooth(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_samples(sawtooth: Sawtooth):
    for genome, score in __SAMPLES:
        solution = Solution(np.array(genome))
        pred_score = sawtooth.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
