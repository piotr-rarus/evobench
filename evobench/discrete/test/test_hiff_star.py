from pytest import fixture

from ..hiff_star import HiffStar

__SAMPLES = [
    ([1] * 24, 36),
    ([0] * 24, 36)
]


@fixture
def hiff_star() -> HiffStar:
    return HiffStar(blocks=[8, 8, 8])


def test_fitness(hiff_star: HiffStar, helpers):
    helpers.check_samples(__SAMPLES, hiff_star)
