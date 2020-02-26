from pytest import fixture

from evobench.util import check_samples

from ..trap import Trap

__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 9),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 12),
    ([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 0),
]


@fixture
def trap() -> Trap:
    return Trap(blocks=[4, 4, 4])


def test_samples(trap: Trap):
    check_samples(__SAMPLES, trap)


def test_global_opt(trap: Trap):
    assert isinstance(trap.global_opt, float)
    assert trap.global_opt == 12


def test_as_dict(trap: Trap):
    assert isinstance(trap.as_dict, dict)
