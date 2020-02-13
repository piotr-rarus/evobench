import numpy as np

from evobench.model import Population, Solution

from ..step_bimodal import StepBimodal

__BLOCK_SIZE = 11
__REPETITIONS = 3
__STEP_SIZE = 2

__GLOBAL_OPTIMUM = 4
__LOCAL_OPTIMUM = 3


def test_global_optima():

    bimodal = StepBimodal(__BLOCK_SIZE, __REPETITIONS, __STEP_SIZE)
    solutions = []

    genome = [0] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)
    solution = Solution(genome)

    solutions.append(solution)

    genome = [1] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)
    solution = Solution(genome)

    solutions.append(solution)
    population = Population(solutions)

    scores = bimodal.evaluate_population(population)

    assert [isinstance(score, int) for score in scores]
    assert [score == __GLOBAL_OPTIMUM for score in scores]
