from typing import List

import numpy as np

from evobench.model.solution import Solution


def shuffle(a: np.ndarray, order: List[int]):
    if a.shape[0] != len(order):
        raise AssertionError('Both array and order must be of the same shape')

    shuffled = np.empty(a.shape, dtype=a.dtype)

    for i, order_index in zip(range(len(order)), order):
        shuffled[i] = a[order_index]

    return shuffled


def deshuffle(a: np.ndarray, order: List[int]):
    if a.shape[0] != len(order):
        raise AssertionError('Both array and order must be of the same shape')

    deshuffled = np.empty(a.shape, dtype=a.dtype)

    for i, order_index in zip(range(len(order)), order):
        deshuffled[order_index] = a[i]

    return deshuffled


def shuffle_solution(solution: Solution, gene_order: List[int]):
    shuffled = shuffle(solution.genome, gene_order)
    return Solution(shuffled)


def deshuffle_solution(solution: Solution, gene_order: List[int]):
    deshuffled = deshuffle(solution.genome, gene_order)
    return Solution(deshuffled)
