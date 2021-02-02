from pytest import fixture

from ..hiff import Hiff

__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1], 32),
    ([1, 1, 0, 0, 1, 0, 0, 1], 6),
]


@fixture
def hiff() -> Hiff:
    return Hiff(blocks=[8])


def test_samples(hiff: Hiff, helpers):
    helpers.check_samples(__SAMPLES, hiff)
