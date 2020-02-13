import numpy as np

from evobench.model import Solution

from ..trap import Trap

__BLOCK_SIZE = 5
__REPETITIONS = 4
__OPTIMUM = float('inf')


def test_optimum():

    trap = Trap(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS,
        overlap_size=0
    )

    genome = [0] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    fitness = trap.evaluate_solution(solution)

    assert isinstance(fitness, float)
    assert fitness == __OPTIMUM
