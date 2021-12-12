import numpy as np
from pytest import fixture

from ..f3 import F3


@fixture(scope="module")
def f3() -> F3:
    return F3()


def test_data_files(f3: F3):
    assert isinstance(f3.xopt, np.ndarray)
    assert f3.xopt.shape == (1000,)


def test_evaluate_solution(f3: F3, helpers):
    helpers.test_evaluate_solution(f3)


def test_evaluate_population(f3: F3, helpers):
    helpers.test_evaluate_population(f3)
