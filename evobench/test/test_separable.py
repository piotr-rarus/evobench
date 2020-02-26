import numpy as np
from pytest import fixture

from evobench.discrete.initialization import Uniform
from evobench.discrete.trap import Trap
from evobench.model import Population

from ..benchmark import Benchmark
from ..initialization import Initialization


__POPULATION_SIZE = 20


@fixture(scope='session')
def initialization() -> Initialization:
    initialization = Uniform(__POPULATION_SIZE)
    return initialization


@fixture(scope='session')
def benchmark() -> Benchmark:
    benchmark = Trap(blocks=[6, 6, 6, 6])
    return benchmark


@fixture(scope='session')
def population(
    initialization: Initialization,
    benchmark: Benchmark
) -> Population:

    population = initialization.initialize_population(benchmark.genome_size)

    return population


def test_evaluate_population(benchmark: Benchmark, population: Population):
    scores = benchmark.evaluate_population(population)

    assert isinstance(scores, np.ndarray)
    assert len(scores.shape) == 1
    assert scores.size == len(population.solutions)
