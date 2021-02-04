import numpy as np
from evobench.model.solution import Solution
from pytest import fixture

from ..trap import Trap

__SAMPLES = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 9),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 12),
    ([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 0),
]


@fixture
def trap() -> Trap:
    return Trap(blocks=[4, 4, 4])


def test_samples(trap: Trap, helpers):
    helpers.check_samples(__SAMPLES, trap)


def test_as_dict(trap: Trap):
    assert isinstance(trap.as_dict, dict)


def test_lower_bound(trap: Trap):
    lower_bound = trap.lower_bound

    assert isinstance(lower_bound, np.ndarray)
    assert lower_bound.size == trap.genome_size


def test_upper_bound(trap: Trap):
    upper_bound = trap.upper_bound

    assert isinstance(upper_bound, np.ndarray)
    assert upper_bound.size == trap.genome_size


def test_bound_range(trap: Trap):
    bound_range = trap.bound_range

    assert isinstance(bound_range, np.ndarray)
    assert bound_range.size == trap.genome_size


def test_random_solution(trap: Trap):
    solution = trap.random_solution()

    assert isinstance(solution, Solution)

    assert isinstance(solution.genome, np.ndarray)
    assert solution.genome.size == trap.genome_size


def test_fix():
    trap = Trap(blocks=[3])
    genome = np.array([0, 1, 2])
    solution = Solution(genome)

    solution = trap.fix(solution)

    assert isinstance(solution, Solution)
    assert isinstance(solution.genome, np.ndarray)

    assert list(solution.genome) == [0, 1, 1]
