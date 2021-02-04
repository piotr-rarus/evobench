from pytest import fixture

from ..step_trap import StepTrap

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 4),
]


@fixture
def step_trap() -> StepTrap:
    return StepTrap(
        blocks=[6, 6],
        step_size=2,
        use_shuffle=True
    )


def test_samples(step_trap: StepTrap, helpers):
    helpers.check_samples(__SAMPLES, step_trap)


def test_as_dict(step_trap: StepTrap):
    assert isinstance(step_trap.as_dict, dict)
