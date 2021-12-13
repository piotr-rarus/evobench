from pytest import fixture

from ..f1 import F1


@fixture(scope="module")
def f1() -> F1:
    return F1()


def test_evaluate_solution(f1: F1, helpers):
    helpers.test_evaluate_solution(f1)


def test_evaluate_population(f1: F1, helpers):
    helpers.test_evaluate_population(f1)
