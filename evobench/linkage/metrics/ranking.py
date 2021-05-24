from typing import List

import numpy as np
from sklearn.metrics import ndcg_score

from evobench.benchmark import Benchmark

# from evobench.util import deshuffle


def mean_reciprocal_rank(
    target_index: int,
    scraps: np.ndarray,
    benchmark: Benchmark,
    k: int = None,
) -> List[float]:

    _assert_scrap_size(scraps, benchmark)

    if k:
        scraps = scraps[:, :k]

    # ! TODO: fix
    # order = benchmark.gene_order[benchmark.gene_order != target_index]
    # order[order > target_index] -= 1
    # scraps = deshuffle(scraps, order)

    relevance = _get_relevant_genes(target_index, scraps, benchmark)

    scores = []
    for ranking in relevance:
        score = _mean_reciprocal_rank(ranking)
        scores.append(score)

    return scores


def mean_average_precision(
    target_index: int,
    scraps: np.ndarray,
    benchmark: Benchmark,
    k: int = None,
) -> List[float]:

    _assert_scrap_size(scraps, benchmark)

    if k:
        scraps = scraps[:, :k]

    relevance = _get_relevant_genes(target_index, scraps, benchmark)

    scores = []
    for ranking in relevance:
        score = _mean_average_precision(ranking)
        scores.append(score)

    return scores


def ndcg(
    target_index: int,
    scraps: np.ndarray,
    interactions: np.ndarray,
    benchmark: Benchmark,
    exp_base: int = 1,
    k: int = None,
) -> List[float]:

    _assert_scrap_size(scraps, benchmark)

    if k:
        scraps = scraps[:, :k]

    assert isinstance(exp_base, int)
    assert scraps.shape == interactions.shape

    exp_base = float(exp_base)
    levels = benchmark.dsm.levels[target_index]
    relevance = levels[scraps].astype(float)
    relevance[relevance <= 0] = np.inf
    relevance -= 1
    relevance = np.power(exp_base, -relevance)

    scores = []
    for interaction, ranking in zip(interactions, relevance):
        score = ndcg_score([interaction], [ranking])
        scores.append(score)

    return scores


def hit_ratio(interactions: np.ndarray) -> float:
    interactions = interactions.astype(bool)
    hits = interactions.sum(axis=1)
    zero_hits = np.sum(hits == 0)

    hit_ratio = (len(interactions) - zero_hits) / len(interactions)
    return hit_ratio


def _assert_scrap_size(scraps: np.ndarray, benchmark: Benchmark):
    assert scraps.shape[1] == benchmark.genome_size - 1


def _get_relevant_genes(
    target_index: int,
    scraps: np.ndarray,
    benchmark: Benchmark
) -> np.ndarray:

    levels = benchmark.dsm.levels[target_index]
    relevance = levels[scraps]
    relevance[relevance == 1] = True  # ? 1st level dependencies
    relevance[relevance != 1] = False
    return relevance.astype(bool)


def _mean_reciprocal_rank(ranking: List[bool]) -> float:
    ranks = np.nonzero(ranking)[0]
    min_index = ranks.min() if ranks.size else float("inf")
    rank = 1 / (min_index + 1)
    return rank


def _mean_average_precision(ranking: List[bool]) -> float:
    ranks = np.nonzero(ranking)[0]

    scores = []

    for index, rank in enumerate(ranks):
        average_precision = (index + 1) / (rank + 1)
        scores.append(average_precision)

    score = 0.

    if scores:
        score = np.mean(scores)

    return score
