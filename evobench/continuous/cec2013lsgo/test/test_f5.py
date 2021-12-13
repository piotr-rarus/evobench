from pytest import fixture

from ..f5 import F5


@fixture(scope="module")
def f5() -> F5:
    return F5()


def test_evaluate_solution(f5: F5, helpers):
    helpers.test_evaluate_solution(f5)


def test_evaluate_population(f5: F5, helpers):
    helpers.test_evaluate_population(f5)
