from pytest import fixture

from evobench.discrete.bimodal import Bimodal
from evobench.util import check_samples

__BLOCK_SIZE = 6
__REPETITIONS = 2

__GLOBAL_OPTIMUM = 6
__LOCAL_OPTIMUM = 4

__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 6),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6),
    ([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1], 3)
]


@fixture
def bimodal() -> Bimodal:
    return Bimodal(__BLOCK_SIZE, __REPETITIONS)


def test_samples(bimodal: Bimodal):
    check_samples(__SAMPLES, bimodal)
