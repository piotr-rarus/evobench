import numpy as np
from evobench.discrete.trap import Trap
from evobench.model import Population
from pytest import fixture

from ..separable import Separable

__POPULATION_SIZE = 20


@fixture(
    scope='module',
    params=[(False, True), (True, False)],
)
def benchmark(request) -> Separable:

    multiprocessing, shuffle = request.param

    return Trap(
        blocks=[6, 6, 6, 6],
        blocks_scaling=[1, 0.75, 0.5, 0.25],
        multiprocessing=multiprocessing,
        use_shuffle=shuffle,
        verbose=1
    )


@fixture(scope='module')
def population(benchmark: Separable) -> Population:
    return benchmark.initialize_population(__POPULATION_SIZE)


def test_evaluate_population(benchmark: Separable, population: Population):
    benchmark.evaluate_population(population)

    fitness = population.fitness

    assert isinstance(fitness, np.ndarray)
    assert len(fitness.shape) == 1
    assert fitness.size == len(population.solutions)


def test_predict(benchmark: Separable, population: Population):
    population_array = population.as_ndarray
    fitness = benchmark.predict(population_array)

    assert isinstance(fitness, np.ndarray)
    assert len(fitness.shape) == 1
    assert fitness.size == len(population.solutions)


def test_true_dsm(benchmark: Separable):
    true_dsm = benchmark.true_dsm

    assert isinstance(true_dsm, np.ndarray)
    assert true_dsm.shape == (benchmark.genome_size, benchmark.genome_size)
