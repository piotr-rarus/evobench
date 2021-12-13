from pytest import fixture

from ..f2 import F2


@fixture(scope="module")
def f2() -> F2:
    return F2()


def test_evaluate_solution(f2: F2, helpers):
    helpers.test_evaluate_solution(f2)


def test_evaluate_population(f2: F2, helpers):
    helpers.test_evaluate_population(f2)
