import numpy as np
from pytest import fixture

from evobench.discrete.isg.config import Config
from evobench.discrete.isg.isg import IsingSpinGlass
from evobench.discrete.isg.spin import Spin
from evobench.model import Solution

__CONFIG_NAME = 'IsingSpinGlass_pm_16_0'


@fixture
def isg() -> IsingSpinGlass:
    return IsingSpinGlass(__CONFIG_NAME, use_shuffle=True)


def test_config(isg: IsingSpinGlass):
    config = isg.config

    assert isinstance(config, Config)

    assert list(config.best_solution) == [
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0
    ]

    assert config.min_energy == -20
    assert isinstance(config.spins, list)
    assert len(config.spins) == 32

    assert all(isinstance(spin, Spin) for spin in config.spins)

    assert isg.genome_size == 16

    assert isinstance(config.as_dict, dict)


def test_as_dict(isg: IsingSpinGlass):

    assert isinstance(isg.as_dict, dict)


def test_best_solution(isg: IsingSpinGlass):

    best_solution = Solution(isg.config.best_solution)
    pred_fitness = isg.evaluate_solution(best_solution)

    assert isinstance(pred_fitness, float)
    assert 0 <= pred_fitness <= 1


def test_true_dsm(isg: IsingSpinGlass):
    dsm = isg.true_dsm

    assert isinstance(dsm, np.ndarray)
    assert dsm.size == isg.genome_size ** 2


def test_lower_bound(isg: IsingSpinGlass):
    lower_bound = isg.lower_bound

    assert isinstance(lower_bound, np.ndarray)
    assert lower_bound.size == isg.genome_size


def test_upper_bound(isg: IsingSpinGlass):
    upper_bound = isg.upper_bound

    assert isinstance(upper_bound, np.ndarray)
    assert upper_bound.size == isg.genome_size


def test_bound_range(isg: IsingSpinGlass):
    bound_range = isg.bound_range

    assert isinstance(bound_range, np.ndarray)
    assert bound_range.size == isg.genome_size


def test_random_solution(isg: IsingSpinGlass):
    solution = isg.random_solution()

    assert isinstance(solution, Solution)

    assert isinstance(solution.genome, np.ndarray)
    assert solution.genome.size == isg.genome_size
