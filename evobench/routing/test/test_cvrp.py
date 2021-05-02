from pathlib import Path

import numpy as np
from evobench.routing.config import Config, Node
from evobench.routing.cvrp import CVRP, Solution
from pytest import fixture

SUITE = ""
INSTANCE = "toy"


@fixture(scope="module")
def cVRP() -> CVRP:
    return CVRP(suite=SUITE, instance=INSTANCE)


def test_config(cVRP: CVRP):
    config = cVRP.config

    assert isinstance(config, Config)
    assert isinstance(config.capacity, float)
    assert isinstance(config.nodes, dict)
    assert isinstance(config.depot, Node)
    assert isinstance(config.global_opt, float)

    assert isinstance(config.best_route, list)
    assert all(isinstance(node, Node) for node in config.best_route)

    assert isinstance(config.target_nodes, list)
    assert all(isinstance(node, Node) for node in config.target_nodes)


def test_initialization(cVRP: CVRP):
    population = cVRP.initialize_population(100)
    fitness = cVRP.evaluate_population(population)

    assert isinstance(fitness, np.ndarray)


def test_best_solution(cVRP: CVRP):

    best_route = cVRP.config.best_route
    genome = [node.id for node in best_route]
    genome = np.array(genome)
    best_solution = Solution(genome)

    fitness = cVRP.evaluate_solution(best_solution)

    assert isinstance(fitness, float)


def test_best_solutions():
    data_a = Path("evobench/routing/data/A")

    for file in data_a.glob("*.vrp"):
        instance = file.stem

        cVRP = CVRP(suite="A", instance=instance)

        best_route = cVRP.config.best_route
        genome = [node.id for node in best_route]
        genome = np.array(genome)
        best_solution = Solution(genome)

        fitness = cVRP.evaluate_solution(best_solution)
        assert isinstance(fitness, float)
