import numpy as np
from pytest import fixture

from ..f2 import F2


@fixture(scope="module")
def f2() -> F2:
    return F2()


def test_data_files(f2: F2):
    assert isinstance(f2.xopt, np.ndarray)
    assert f2.xopt.shape == (1000,)


def test_evaluate_solution(f2: F2, helpers):
    helpers.test_evaluate_solution(f2)


def test_evaluate_population(f2: F2, helpers):
    helpers.test_evaluate_population(f2)
