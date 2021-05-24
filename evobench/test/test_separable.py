import numpy as np
from pytest import fixture

from evobench.discrete.trap import Trap
from evobench.linkage.dsm import DependencyStructureMatrix
from evobench.model import Population

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
    # ! TODO: fix
    # assert benchmark.ffe == len(population.solutions)


def test_dsm(benchmark: Separable):
    assert isinstance(benchmark.dsm, DependencyStructureMatrix)
