from pytest import fixture

from ..f3 import F3


@fixture(scope="module")
def f3() -> F3:
    return F3()


def test_evaluate_solution(f3: F3, helpers):
    helpers.test_evaluate_solution(f3)


def test_evaluate_population(f3: F3, helpers):
    helpers.test_evaluate_population(f3)
