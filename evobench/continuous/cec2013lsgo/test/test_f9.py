from pytest import fixture

from ..f9 import F9


@fixture(scope="module")
def f9() -> F9:
    return F9()


def test_evaluate_solution(f9: F9, helpers):
    helpers.test_evaluate_solution(f9)


def test_evaluate_population(f9: F9, helpers):
    helpers.test_evaluate_population(f9)


def test_dsm(f9: F9, helpers):
    helpers.test_dsm(f9)
