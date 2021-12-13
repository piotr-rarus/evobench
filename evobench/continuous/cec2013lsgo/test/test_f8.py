from pytest import fixture

from ..f8 import F8


@fixture(scope="module")
def f8() -> F8:
    return F8()


def test_evaluate_solution(f8: F8, helpers):
    helpers.test_evaluate_solution(f8)


def test_evaluate_population(f8: F8, helpers):
    helpers.test_evaluate_population(f8)
