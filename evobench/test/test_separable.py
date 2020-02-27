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
    return Uniform(__POPULATION_SIZE)


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
def population(
    initialization: Initialization,
    benchmark: Benchmark
) -> Population:

    return initialization.initialize_population(benchmark.genome_size)


def test_evaluate_population(benchmark: Benchmark, population: Population):
    scores = benchmark.evaluate_population(population)

    assert isinstance(scores, np.ndarray)
    assert len(scores.shape) == 1
    assert scores.size == len(population.solutions)
