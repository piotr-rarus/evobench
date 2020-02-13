from pytest import fixture

from evobench.util import check_samples

from ..sawtooth import Sawtooth

__BLOCK_SIZE = 5
__REPETITIONS = 2

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0),
    ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1),
]


@fixture
def sawtooth() -> Sawtooth:
    return Sawtooth(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)


def test_samples(sawtooth: Sawtooth):
    check_samples(__SAMPLES, sawtooth)
