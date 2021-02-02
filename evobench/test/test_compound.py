import numpy as np
from evobench.continuous.trap import Trap as ContinuousTrap
from evobench.discrete.trap import Trap as DiscreteTrap
from evobench.model import Population
from pytest import fixture

from ..compound import CompoundBenchmark

__POPULATION_SIZE = 20


@fixture(
    scope='module',
    params=[(False, True), (True, False)],
)
def benchmark(request) -> CompoundBenchmark:

    multiprocessing, shuffle = request.param

    return CompoundBenchmark(
        benchmarks=[
            DiscreteTrap(blocks=[5]),
            ContinuousTrap(blocks=[5])
        ],
        multiprocessing=multiprocessing,
        use_shuffle=shuffle
    )


@fixture(scope='module')
def population(benchmark: CompoundBenchmark) -> Population:
    return benchmark.initialize_population(__POPULATION_SIZE)


def test_evaluate_population(
    benchmark: CompoundBenchmark,
    population: Population
):
    benchmark.evaluate_population(population)

    fitness = population.fitness

    assert isinstance(fitness, np.ndarray)
    assert len(fitness.shape) == 1
    assert fitness.size == len(population.solutions)


def test_true_dsm(benchmark: CompoundBenchmark):
    true_dsm = benchmark.true_dsm

    assert isinstance(true_dsm, np.ndarray)
    assert true_dsm.shape == (benchmark.genome_size, benchmark.genome_size)


def test_lower_bound(benchmark: CompoundBenchmark):
    lower_bound = benchmark.lower_bound
    assert isinstance(lower_bound, np.ndarray)
    assert len(lower_bound) == benchmark.genome_size


def test_upper_bound(benchmark: CompoundBenchmark):
    upper_bound = benchmark.upper_bound
    assert isinstance(upper_bound, np.ndarray)
    assert len(upper_bound) == benchmark.genome_size


def test_as_dict(benchmark: CompoundBenchmark):
    as_dict = benchmark.as_dict
    assert isinstance(as_dict, dict)
