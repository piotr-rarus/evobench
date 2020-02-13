import numpy as np

from evobench.model import Solution

from ..sawtooth import Sawtooth

__BLOCK_SIZE = 5
__REPETITIONS = 4
__MINIMA = 0


def test_minima():

    sawtooth = Sawtooth(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS,
        overlap_size=0
    )

    genome = [0] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    fitness = sawtooth.evaluate_solution(solution)

    assert isinstance(fitness, float)
    assert fitness == __MINIMA
