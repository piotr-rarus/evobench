from typing import List

import numpy as np
from pytest import fixture

from evobench.discrete.trap import Trap
from evobench.model import Population
from evobench.util import dsm_fill_quality

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
        multiprocessing=multiprocessing,
        shuffle=shuffle
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


def test_true_dsm(benchmark: Separable):
    true_dsm = benchmark.true_dsm

    assert isinstance(true_dsm, np.ndarray)
    assert true_dsm.shape == (benchmark.genome_size, benchmark.genome_size)


def test_dsm_fill_quality():
    benchmark = Trap(blocks=[2, 3])

    pred_dsm = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ]

    pred_dsm = np.array(pred_dsm)

    fill_quality = dsm_fill_quality(pred_dsm, benchmark.true_dsm)

    assert isinstance(fill_quality, List)
    assert len(fill_quality) == 5

    assert fill_quality == [1, 1, 0, 0.5, 0.5]
