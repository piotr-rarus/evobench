from pytest import fixture

from ..f14 import F14


@fixture(scope="module")
def f14() -> F14:
    return F14()


def test_evaluate_solution(f14: F14, helpers):
    helpers.test_evaluate_solution(f14)


def test_evaluate_population(f14: F14, helpers):
    helpers.test_evaluate_population(f14)
