from evobench.discrete.isg.isg import IsingSpinGlass
from pytest import fixture
from evobench.discrete.isg.config import Config
from evobench.discrete.isg.spin import Spin

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

    assert config.global_optimum == -20
    assert isinstance(config.spins, list)
    assert len(config.spins) == 32

    assert all(isinstance(spin, Spin) for spin in config.spins)

    assert isg.genome_size == 16
