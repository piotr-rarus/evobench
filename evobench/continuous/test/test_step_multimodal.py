import numpy as np

from evobench.model import Solution

from ..step_multimodal import StepMultimodal

__BLOCK_SIZE = 5
__REPETITIONS = 4
__OPTIMUM = 0


def test_optimum():

    step_multimodal = StepMultimodal(
        block_size=__BLOCK_SIZE,
        repetitions=__REPETITIONS,
        step_size=0.1,
        overlap_size=0
    )

    genome = [0] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    fitness = step_multimodal.evaluate_solution(solution)

    assert isinstance(fitness, float)
    assert fitness == __OPTIMUM
