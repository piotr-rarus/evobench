from pytest import fixture

from ..f4 import F4


@fixture(scope="module")
def f4() -> F4:
    return F4()


def test_evaluate_solution(f4: F4, helpers):
    helpers.test_evaluate_solution(f4)


def test_evaluate_population(f4: F4, helpers):
    helpers.test_evaluate_population(f4)


def test_dsm(f4: F4, helpers):
    helpers.test_dsm(f4)
