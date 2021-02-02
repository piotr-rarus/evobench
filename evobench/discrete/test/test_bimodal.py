from evobench.discrete.bimodal import Bimodal
from pytest import fixture

__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 6),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6),
    ([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1], 3)
]


@fixture
def bimodal() -> Bimodal:
    return Bimodal(blocks=[6, 6])


def test_samples(bimodal: Bimodal, helpers):
    helpers.check_samples(__SAMPLES, bimodal)
