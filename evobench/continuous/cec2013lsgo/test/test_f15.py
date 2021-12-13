from pytest import fixture

from ..f15 import F15


@fixture(scope="module")
def f15() -> F15:
    return F15()


def test_evaluate_solution(f15: F15, helpers):
    helpers.test_evaluate_solution(f15)


def test_evaluate_population(f15: F15, helpers):
    helpers.test_evaluate_population(f15)


def test_dsm(f15: F15, helpers):
    helpers.test_dsm(f15)
