import numpy as np
from pytest import fixture

from evobench.discrete.isg.config import Config
from evobench.discrete.isg.isg import IsingSpinGlass
from evobench.discrete.isg.spin import Spin
from evobench.model import Solution

__CONFIG_NAME = 'IsingSpinGlass_pm_16_0'


@fixture
def isg() -> IsingSpinGlass:
    return IsingSpinGlass(__CONFIG_NAME)


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
    pred_score = isg.evaluate_solution(best_solution)

    assert isinstance(pred_score, float)
    assert 0 <= pred_score <= 1


def test_dsm(isg: IsingSpinGlass):
    dsm = isg.dsm

    assert isinstance(dsm, np.ndarray)
    assert dsm.size == isg.genome_size ** 2
