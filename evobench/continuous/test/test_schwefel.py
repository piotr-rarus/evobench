from pytest import fixture

from evobench.util import check_samples

from ..schwefel import Schwefel

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0], 3770.8460999999998),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1], 3763.272861136728),
]


@fixture
def schwefel() -> Schwefel:
    return Schwefel(blocks=[5, 4])


def test_samples(schwefel: Schwefel):
    check_samples(__SAMPLES, schwefel)
