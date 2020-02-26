from pytest import fixture

from evobench.util import check_samples

from ..trap import Trap


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float('inf')),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10),
    ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 4),
]


@fixture
def trap() -> Trap:
    return Trap(blocks=[5, 5])


def test_samples(trap: Trap):
    check_samples(__SAMPLES, trap)


def test_global_opt(trap: Trap):
    assert isinstance(trap.global_opt, float)
    assert trap.global_opt == float('inf')
