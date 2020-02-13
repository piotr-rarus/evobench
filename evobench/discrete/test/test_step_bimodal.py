import numpy as np
from pytest import fixture

from evobench.discrete.step_bimodal import StepBimodal
from evobench.model import Solution

__BLOCK_SIZE = 11
__REPETITIONS = 1
__STEP_SIZE = 2


SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3),
    ([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 2),
    ([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 1),

]


@fixture
def step_bimodal() -> StepBimodal:
    return StepBimodal(__BLOCK_SIZE, __REPETITIONS, __STEP_SIZE)


def test_samples(step_bimodal: StepBimodal):
    for genome, score in SAMPLES:
        solution = Solution(np.array(genome))
        pred_score = step_bimodal.evaluate_solution(solution)

        assert isinstance(pred_score, float)
        assert pred_score == score
