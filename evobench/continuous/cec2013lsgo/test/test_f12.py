from pytest import fixture

from ..f12 import F12


@fixture(scope="module")
def f12() -> F12:
    return F12()


def test_evaluate_solution(f12: F12, helpers):
    helpers.test_evaluate_solution(f12)


def test_evaluate_population(f12: F12, helpers):
    helpers.test_evaluate_population(f12)
