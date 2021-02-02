import numpy as np
from pytest import fixture

from ..solution import Solution


@fixture(scope='module')
def solution() -> Solution:
    genome = np.array([1, 2, 3, 4, 5])
    return Solution(genome)


def test_hash(solution: Solution):
    h = solution.__hash__
    assert isinstance(h, str)
