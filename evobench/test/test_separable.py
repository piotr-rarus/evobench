import numpy as np
from pytest import fixture

from evobench.discrete.trap import Trap
from evobench.model import Population

from ..benchmark import Benchmark

__POPULATION_SIZE = 20


@fixture(
    scope='module',
    params=[(False, True), (True, False)],
)
def benchmark(request) -> Benchmark:

    multiprocessing, shuffle = request.param

    return Trap(
        blocks=[6, 6, 6, 6],
        multiprocessing=multiprocessing,
        shuffle=shuffle
    )


@fixture(scope='module')
def population(benchmark: Benchmark) -> Population:
    return benchmark.initialize_population(__POPULATION_SIZE)


def test_evaluate_population(benchmark: Benchmark, population: Population):
    benchmark.evaluate_population(population)

    fitness = population.fitness

    assert isinstance(fitness, np.ndarray)
    assert len(fitness.shape) == 1
    assert fitness.size == len(population.solutions)
