import numpy as np

from evobench.model import Solution

from ..step_trap import StepTrap

__BLOCK_SIZE = 8
__REPETITIONS = 3
__STEP_SIZE = 2

__GLOBAL_OPTIMUM = 12
__LOCAL_OPTIMUM = 9


def test_global_optimum():

    trap = StepTrap(__BLOCK_SIZE, __REPETITIONS, __STEP_SIZE)

    genome = [0] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    score = trap.evaluate_solution(solution)

    assert isinstance(score, float)
    assert score == __GLOBAL_OPTIMUM


def test_local_optima():

    trap = StepTrap(__BLOCK_SIZE, __REPETITIONS, __STEP_SIZE)

    genome = [1] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    score = trap.evaluate_solution(solution)

    assert isinstance(score, float)
    assert score == __LOCAL_OPTIMUM
