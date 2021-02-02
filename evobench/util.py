from typing import List

import numpy as np

from evobench.model.solution import Solution


def shuffle(a: np.ndarray, order: np.ndarray) -> np.ndarray:
    if a.shape != order.shape:
        raise AssertionError('Both array and order must be of the same shape')

    shuffled = a[order]
    return shuffled


def deshuffle(a: np.ndarray, order: np.ndarray) -> np.ndarray:
    if a.shape != order.shape:
        raise AssertionError('Both array and order must be of the same shape')

    deshuffled = np.empty(a.shape, dtype=a.dtype)
    deshuffled[order] = a

    return deshuffled


def shuffle_solution(solution: Solution, gene_order: List[int]) -> Solution:
    shuffled = shuffle(solution.genome, gene_order)
    return Solution(shuffled)


def deshuffle_solution(solution: Solution, gene_order: List[int]) -> Solution:
    deshuffled = deshuffle(solution.genome, gene_order)
    return Solution(deshuffled)
