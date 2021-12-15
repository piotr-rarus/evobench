from pytest import fixture

from ..f6 import F6


@fixture(scope="module")
def f6() -> F6:
    return F6()


def test_evaluate_solution(f6: F6, helpers):
    helpers.test_evaluate_solution(f6)


def test_evaluate_population(f6: F6, helpers):
    helpers.test_evaluate_population(f6)


def test_dsm(f6: F6, helpers):
    helpers.test_dsm(f6)
