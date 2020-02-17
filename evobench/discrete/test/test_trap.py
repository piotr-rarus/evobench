from pytest import fixture

from evobench.util import check_samples

from ..trap import Trap

__BLOCK_SIZE = 4
__REPETITIONS = 3

__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 9),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 12),
    ([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 0),
]


@fixture
def trap() -> Trap:
    return Trap(__BLOCK_SIZE, __REPETITIONS)


def test_samples(trap: Trap):
    check_samples(__SAMPLES, trap)


def test_as_dict():
    trap = Trap(__BLOCK_SIZE, __REPETITIONS)

    assert isinstance(trap.as_dict, dict)
