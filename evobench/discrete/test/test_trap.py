import numpy as np

from evobench.model import Solution

from ..trap import Trap

__BLOCK_SIZE = 8
__REPETITIONS = 3

__GLOBAL_OPTIMUM = 24
__LOCAL_OPTIMUM = 21


def test_global_optimum():

    trap = Trap(__BLOCK_SIZE, __REPETITIONS)

    genome = [0] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    score = trap.evaluate_solution(solution)

    assert isinstance(score, float)
    assert score == __GLOBAL_OPTIMUM


def test_local_optimum():

    trap = Trap(__BLOCK_SIZE, __REPETITIONS)

    genome = [1] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    score = trap.evaluate_solution(solution)

    assert isinstance(score, float)
    assert score == __LOCAL_OPTIMUM
