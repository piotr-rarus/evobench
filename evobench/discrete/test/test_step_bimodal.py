from pytest import fixture

from evobench.discrete.step_bimodal import StepBimodal
from evobench.util import check_samples

__BLOCK_SIZE = 11
__REPETITIONS = 1
__STEP_SIZE = 2


__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3),
    ([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 2),
    ([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 1),

]


@fixture
def step_bimodal() -> StepBimodal:
    return StepBimodal(__BLOCK_SIZE, __REPETITIONS, __STEP_SIZE)


def test_samples(step_bimodal: StepBimodal):
    check_samples(__SAMPLES, step_bimodal)
