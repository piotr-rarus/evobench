from pathlib import Path

from pytest import fixture

from evobench.routing import parser
from evobench.routing.config import Config, Node


@fixture(scope="module")
def config() -> Config:
    data_dir = Path(__file__).parent.parent.joinpath("data")
    return parser.load(
        instance_path=data_dir.joinpath("toy.vrp"),
        solution_path=data_dir.joinpath("toy.sol")
    )


def test_config(config: Config):
    assert isinstance(config, Config)
    assert isinstance(config.capacity, float)
    assert isinstance(config.nodes, dict)
    assert isinstance(config.depot, Node)
    assert isinstance(config.global_opt, float)

    assert isinstance(config.best_route, list)
    assert all(isinstance(node, Node) for node in config.best_route)

    assert isinstance(config.target_nodes, list)
    assert all(isinstance(node, Node) for node in config.target_nodes)
