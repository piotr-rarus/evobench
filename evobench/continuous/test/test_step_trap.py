from pytest import fixture

from ..step_trap import StepTrap

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float('inf')),
    ([1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0], 3.0),
    ([1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 4.0)
]


@fixture
def step_trap() -> StepTrap:
    return StepTrap(blocks=[5, 5], step_size=0.1)


def test_samples(step_trap: StepTrap, helpers):
    helpers.check_samples(__SAMPLES, step_trap)


def test_as_dict(step_trap: StepTrap):
    as_dict = step_trap.as_dict
    assert isinstance(as_dict, dict)
