from pytest import fixture
from evobench.util import check_samples

from ..hiff import Hiff

__BLOCK_SIZE = 8
__REPETITIONS = 1


__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1], 12),
    ([1, 1, 0, 0, 1, 0, 0, 1], 1),
]


@fixture
def hiff() -> Hiff:
    return Hiff(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_samples(hiff: Hiff):
    check_samples(__SAMPLES, hiff)
