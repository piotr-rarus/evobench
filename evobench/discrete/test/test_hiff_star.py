from pytest import fixture
from evobench.util import check_samples

from ..hiff_star import HiffStar

__BLOCK_SIZE = 8
__REPETITIONS = 3
__GLOBAL_OPTIMUM = 36

__SAMPLES = [
    ([1] * __BLOCK_SIZE * __REPETITIONS, __GLOBAL_OPTIMUM),
    ([0] * __BLOCK_SIZE * __REPETITIONS, __GLOBAL_OPTIMUM)
]


@fixture
def hiff_star() -> HiffStar:
    return HiffStar(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_scores(hiff_star: HiffStar):
    check_samples(__SAMPLES, hiff_star)
