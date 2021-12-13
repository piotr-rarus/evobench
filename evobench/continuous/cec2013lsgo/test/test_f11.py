from pytest import fixture

from ..f11 import F11


@fixture(scope="module")
def f11() -> F11:
    return F11()


def test_evaluate_solution(f11: F11, helpers):
    helpers.test_evaluate_solution(f11)


def test_evaluate_population(f11: F11, helpers):
    helpers.test_evaluate_population(f11)
