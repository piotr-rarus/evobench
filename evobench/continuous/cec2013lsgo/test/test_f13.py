from pytest import fixture

from ..f13 import F13


@fixture(scope="module")
def f13() -> F13:
    return F13()


def test_evaluate_solution(f13: F13, helpers):
    helpers.test_evaluate_solution(f13)


def test_evaluate_population(f13: F13, helpers):
    helpers.test_evaluate_population(f13)


def test_dsm(f13: F13, helpers):
    helpers.test_dsm(f13)
