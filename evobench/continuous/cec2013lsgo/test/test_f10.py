from pytest import fixture

from ..f10 import F10


@fixture(scope="module")
def f10() -> F10:
    return F10()


def test_evaluate_solution(f10: F10, helpers):
    helpers.test_evaluate_solution(f10)


def test_evaluate_population(f10: F10, helpers):
    helpers.test_evaluate_population(f10)
