from pytest import fixture

from evobench.util import check_samples

from ..trap import Trap

__BLOCK_SIZE = 5
__REPETITIONS = 2


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float('inf')),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10),
    ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 4),
]


@fixture
def trap() -> Trap:
    return Trap(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS
    )


def test_samples(trap: Trap):
    check_samples(__SAMPLES, trap)
