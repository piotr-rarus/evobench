from pytest import fixture

from evobench.util import check_samples

from ..rastrigin import Rastrigin

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1], 9),
]


@fixture
def rastrigin() -> Rastrigin:
    return Rastrigin(blocks=[5, 4])


def test_samples(rastrigin: Rastrigin):
    check_samples(__SAMPLES, rastrigin)
