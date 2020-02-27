from pytest import fixture
from evobench.util import check_samples

from ..hiff_star import HiffStar


__SAMPLES = [
    ([1] * 24, 36),
    ([0] * 24, 36)
]


@fixture
def hiff_star() -> HiffStar:
    return HiffStar(blocks=[8, 8, 8])


def test_scores(hiff_star: HiffStar):
    check_samples(__SAMPLES, hiff_star)


def test_global_opt(hiff_star: HiffStar):
    assert isinstance(hiff_star.global_opt, float)
    assert hiff_star.global_opt == 36
