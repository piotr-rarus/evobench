import numpy as np
from pytest import fixture

from evobench.model import Solution

from ..step_multimodal import StepMultimodal

__BLOCK_SIZE = 5
__REPETITIONS = 2

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def step_multimodal() -> StepMultimodal:
    return StepMultimodal(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS,
        step_size=0.1,
    )


def test_samples(step_multimodal: StepMultimodal):
    for genome, score in __SAMPLES:
        solution = Solution(np.array(genome))
        pred_score = step_multimodal.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
