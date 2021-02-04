import numpy as np
from pytest import fixture

from ..population import Population
from ..solution import Solution


@fixture
def population() -> Population:
    solutions = []

    solutions.append(Solution(np.array([1, 2, 1, 4, 5])))
    solutions.append(Solution(np.array([2, 2, 3, 4, 5])))
    solutions.append(Solution(np.array([4, 3, 5, 4, 5])))

    return Population(solutions)


def test_upper_bound(population: Population):
    upper_bound = np.array([4, 3, 5, 4, 5])

    assert np.array_equal(population.upper_bound, upper_bound)


def test_lower_bound(population: Population):
    lower_bound = [1, 2, 1, 4, 5]

    assert np.array_equal(population.lower_bound, lower_bound)


def test_as_ndarray(population: Population):
    assert isinstance(population.as_ndarray, np.ndarray)


def test_size(population: Population):
    size = population.size
    assert isinstance(size, int)
    assert size == 3
