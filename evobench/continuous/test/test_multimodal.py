from pytest import fixture

from evobench.util import check_samples

from ..multimodal import Multimodal

__BLOCK_SIZE = 5
__REPETITIONS = 2


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def multimodal() -> Multimodal:
    return Multimodal(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_samples(multimodal: Multimodal):
    check_samples(__SAMPLES, multimodal)
