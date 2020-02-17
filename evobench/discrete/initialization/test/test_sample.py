from typing import Dict

from numpy import ndarray

from evobench.model import Population, Solution

from ..uniform import Uniform

__GENOME_SIZE = 10
__POPULATION_SIZE = 1000


def test_initialize_population():
    uniform_sample = Uniform(__POPULATION_SIZE)

    population = uniform_sample.initialize_population(__GENOME_SIZE)

    assert isinstance(population, Population)

    assert all(
        isinstance(solution, Solution) for
        solution in population.solutions
    )

    assert len(population.solutions) == __POPULATION_SIZE

    assert all(
        isinstance(solution.genome, ndarray) for
        solution in population.solutions
    )

    assert all(
        solution.genome.size == __GENOME_SIZE for
        solution in population.solutions
    )


def test_as_dict():
    uniform_sample = Uniform(__POPULATION_SIZE)
    as_dict = uniform_sample.as_dict

    assert isinstance(as_dict, Dict)
