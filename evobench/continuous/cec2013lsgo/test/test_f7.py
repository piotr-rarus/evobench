from pytest import fixture

from ..f7 import F7


@fixture(scope="module")
def f7() -> F7:
    return F7()


def test_evaluate_solution(f7: F7, helpers):
    helpers.test_evaluate_solution(f7)


def test_evaluate_population(f7: F7, helpers):
    helpers.test_evaluate_population(f7)
