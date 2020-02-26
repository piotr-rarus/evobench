from pytest import fixture

from evobench.util import check_samples

from ..sawtooth import Sawtooth

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0),
    ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1),
]


@fixture
def sawtooth() -> Sawtooth:
    return Sawtooth(blocks=[5, 5])


def test_samples(sawtooth: Sawtooth):
    check_samples(__SAMPLES, sawtooth)


def test_global_opt(sawtooth: Sawtooth):
    assert isinstance(sawtooth.global_opt, float)
    assert sawtooth.global_opt == 2
