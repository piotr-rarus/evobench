from pytest import fixture

from ..multimodal import Multimodal


__SAMPLES = [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
]


@fixture
def multimodal() -> Multimodal:
    return Multimodal(blocks=[5, 5])


def test_samples(multimodal: Multimodal, helpers):
    helpers.check_samples(__SAMPLES, multimodal)


def test_as_dict(multimodal: Multimodal):
    as_dict = multimodal.as_dict
    assert isinstance(as_dict, dict)
