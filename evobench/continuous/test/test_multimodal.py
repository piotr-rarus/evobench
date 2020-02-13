import numpy as np

from evobench.model import Solution

from ..multimodal import Multimodal

__BLOCK_SIZE = 5
__REPETITIONS = 4
__MIN = 0


def test_min():

    multimodal = Multimodal(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS,
        overlap_size=0
    )

    genome = [0] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    fitness = multimodal.evaluate_solution(solution)

    assert isinstance(fitness, float)
    assert fitness == __MIN
