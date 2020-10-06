from pytest import fixture

from evobench.util import check_samples

from ..sphere import Sphere

__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    ([1, 1, 1, 1, 1, 1, 1, 1, 1], 9),
]


@fixture
def sphere() -> Sphere:
    return Sphere(blocks=[5, 4])


def test_samples(sphere: Sphere):
    check_samples(__SAMPLES, sphere)
