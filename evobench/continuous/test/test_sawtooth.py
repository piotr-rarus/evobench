from pytest import fixture

from ..sawtooth import Sawtooth

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0),
    ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1),
]


@fixture
def sawtooth() -> Sawtooth:
    return Sawtooth(blocks=[5, 5])


def test_samples(sawtooth: Sawtooth, helpers):
    helpers.check_samples(__SAMPLES, sawtooth)
