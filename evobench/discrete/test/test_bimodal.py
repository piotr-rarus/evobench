import numpy as np

from evobench.discrete.bimodal import Bimodal
from evobench.model import Population, Solution

__BLOCK_SIZE = 6
__REPETITIONS = 3

__GLOBAL_OPTIMUM = 9
__LOCAL_OPTIMUM = 6


def test_global_optima():

    bimodal = Bimodal(__BLOCK_SIZE, __REPETITIONS)
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
