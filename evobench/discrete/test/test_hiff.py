import numpy as np

from evobench.model import Solution

from ..hiff import Hiff

__BLOCK_SIZE = 8
__REPETITIONS = 3
__GLOBAL_OPTIMUM = 36


def test_global_optimum():

    hiff = Hiff(block_size=__BLOCK_SIZE, repetitions=__REPETITIONS)

    genome = [1] * __BLOCK_SIZE * __REPETITIONS
    genome = np.array(genome)

    solution = Solution(genome)
    score = hiff.evaluate_solution(solution)

    assert score == __GLOBAL_OPTIMUM


def test_solution():

    hiff = Hiff(block_size=8, repetitions=1)

    genome = [1, 1, 0, 0, 1, 0, 0, 1]
    genome = np.array(genome)

    solution = Solution(genome)
    score = hiff.evaluate_solution(solution)

    assert score == 1
