from pytest import fixture

from evobench.util import check_samples

from ..step_trap import StepTrap


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 4),
]


@fixture
def step_trap() -> StepTrap:
    return StepTrap(
        blocks=[6, 6],
        step_size=2
    )


def test_samples(step_trap: StepTrap):
    check_samples(__SAMPLES, step_trap)


def test_global_opt(step_trap: StepTrap):
    assert isinstance(step_trap.global_opt, float)
    assert step_trap.global_opt == 6


def test_as_dict(step_trap: StepTrap):
    assert isinstance(step_trap.as_dict, dict)
