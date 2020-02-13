import numpy as np
from pytest import fixture

from evobench.model import Solution

from ..trap import Trap

__BLOCK_SIZE = 5
__REPETITIONS = 2


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float('inf')),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10),
    ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 4),
]


@fixture
def trap() -> Trap:
    return Trap(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS
    )


def test_samples(trap: Trap):
    for genome, score in __SAMPLES:
        solution = Solution(np.array(genome))
        pred_score = trap.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
