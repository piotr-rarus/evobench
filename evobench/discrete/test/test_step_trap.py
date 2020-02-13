from pytest import fixture

from evobench.util import check_samples

from ..step_trap import StepTrap

__BLOCK_SIZE = 6
__REPETITIONS = 2
__STEP_SIZE = 2

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 4),
]


@fixture
def step_trap() -> StepTrap:
    return StepTrap(__BLOCK_SIZE, __REPETITIONS, __STEP_SIZE)


def test_samples(step_trap: StepTrap):
    check_samples(__SAMPLES, step_trap)


def test_as_dict(step_trap: StepTrap):
    assert isinstance(step_trap.as_dict, dict)
