from pytest import fixture

from evobench.util import check_samples

from ..multimodal import Multimodal


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def multimodal() -> Multimodal:
    return Multimodal(blocks=[5, 5])


def test_samples(multimodal: Multimodal):
    check_samples(__SAMPLES, multimodal)


def test_global_opt(multimodal: Multimodal):
    assert isinstance(multimodal.global_opt, float)
    assert multimodal.global_opt == 2
