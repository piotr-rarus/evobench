import numpy as np
from pytest import fixture

from evobench.model.population import Population
from evobench.model.solution import Solution


@fixture
def population() -> Population:
    solutions = []

    solutions.append(Solution(np.array([1, 2, 1, 4, 5])))
    solutions.append(Solution(np.array([2, 2, 3, 4, 5])))
    solutions.append(Solution(np.array([4, 3, 5, 4, 5])))

    return Population(solutions)


def test_as_ndarray(population: Population):
    assert isinstance(population.as_ndarray, np.ndarray)
