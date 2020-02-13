from numpy import ndarray

from evobench.model import Population, Solution

from ..complete import Complete

__GENOME_SIZE = 6


def test_initialize_population():
    complete = Complete(population_size=0)

    population = complete.initialize_population(__GENOME_SIZE)

    assert isinstance(population, Population)

    assert all(
        isinstance(solution, Solution) for
        solution in population.solutions
    )

    expected_population_size = 2 ** __GENOME_SIZE

    assert len(population.solutions) == expected_population_size

    assert all(
        isinstance(solution.genome, ndarray) for
        solution in population.solutions
    )

    assert all(
        solution.genome.size == __GENOME_SIZE for
        solution in population.solutions
    )
