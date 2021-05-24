import numpy as np
from pytest import fixture

from evobench.discrete.trap import Trap

from ..ranking import hit_ratio, mean_average_precision, mean_reciprocal_rank, ndcg


@fixture(scope="module")
def trap() -> Trap:
    return Trap(blocks=[3] * 2)


@fixture(scope="module")
def scraps() -> np.ndarray:
    scraps = [
        [2, 1, 3, 4, 5],
        [2, 3, 1, 4, 5],
        [3, 5, 4, 1, 2],
        [5, 3, 1, 2, 4]
    ]

    return np.array(scraps)


@fixture(scope="module")
def interactions() -> np.ndarray:
    interactions = [
        [0.9, 0.8, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.3, 0.2, 0.1],
        [0.] * 5,
    ]

    return np.array(interactions)


def test_mean_reciprocal_rank(scraps: np.ndarray, trap: Trap):
    target_index = 0
    scores = mean_reciprocal_rank(target_index, scraps, trap)

    assert isinstance(scores, list)
    assert len(scores) == len(scraps)
    assert all(isinstance(score, float) for score in scores)
    assert all(0 <= score <= 1 for score in scores)
    assert scores == [1, 1, 0.25, 1/3]


def test_mean_average_precision(scraps: np.ndarray, trap: Trap):
    target_index = 0
    scores = mean_average_precision(target_index, scraps, trap)

    assert isinstance(scores, list)
    assert len(scores) == len(scraps)
    assert all(isinstance(score, float) for score in scores)
    assert all(0 <= score <= 1 for score in scores)

    scores = [round(score, 2) for score in scores]
    assert scores == [1, 0.83, 0.32, 0.42]


def test_ndcg(scraps: np.ndarray, interactions: np.ndarray, trap: Trap):
    target_index = 0
    scores = ndcg(target_index, scraps, interactions, trap, exp_base=2)

    assert isinstance(scores, list)
    assert len(scores) == len(scraps)
    assert all(isinstance(score, float) for score in scores)
    assert all(0 <= score <= 1 for score in scores)

    scores = [round(score, 2) for score in scores]

    assert scores == [0.98, 0.87, 0.67, 0]


def test_hit_ratio(interactions: np.ndarray):
    score = hit_ratio(interactions)

    assert isinstance(score, float)
    assert score == 0.75
