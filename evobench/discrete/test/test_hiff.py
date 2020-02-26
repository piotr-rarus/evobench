from pytest import fixture
from evobench.util import check_samples

from ..hiff import Hiff


__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1], 32),
    ([1, 1, 0, 0, 1, 0, 0, 1], 6),
]


@fixture
def hiff() -> Hiff:
    return Hiff(blocks=[8])


def test_samples(hiff: Hiff):
    check_samples(__SAMPLES, hiff)


def test_global_opt(hiff: Hiff):
    assert isinstance(hiff.global_opt, float)
    assert hiff.global_opt == 32
