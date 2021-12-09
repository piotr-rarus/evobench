import numpy as np
from pytest import fixture

from ..f1 import F1


@fixture(scope="module")
def f1() -> F1:
    return F1()


def test_xopt(f1: F1):
    assert isinstance(f1.xopt, np.ndarray)
    assert f1.xopt.shape == (1000,)


def test_evaluate_solution(f1: F1, helpers):
    helpers.test_evaluate_solution(f1)


def test_evaluate_population(f1: F1, helpers):
    helpers.test_evaluate_population(f1)
