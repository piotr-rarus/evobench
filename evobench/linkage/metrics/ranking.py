from typing import List

import numpy as np

from evobench.benchmark import Benchmark

# from evobench.util import deshuffle


def mean_reciprocal_rank(
    target_index: int,
    scraps: np.ndarray,
    benchmark: Benchmark,
    k: int = None,
) -> np.ndarray:

    _assert_scrap_size(scraps, benchmark)

    if k:
        scraps = scraps[:, :k]

    # ! TODO: fix
    # order = benchmark.gene_order[benchmark.gene_order != target_index]
    # order[order > target_index] -= 1
    # scraps = deshuffle(scraps, order)

    relevance = _get_relevant_genes(target_index, scraps, benchmark)
    scores = [_mean_reciprocal_rank(ranking) for ranking in relevance]
    return np.array(scores)


def mean_average_precision(
    target_index: int,
    scraps: np.ndarray,
    benchmark: Benchmark,
    k: int = None,
) -> np.ndarray:

    _assert_scrap_size(scraps, benchmark)

    if k:
        scraps = scraps[:, :k]

    relevance = _get_relevant_genes(target_index, scraps, benchmark)
    scores = [_mean_average_precision(ranking) for ranking in relevance]
    return np.array(scores)


def ndcg(
    target_index: int,
    scraps: np.ndarray,
    benchmark: Benchmark,
    k: int = None,
) -> np.ndarray:

    _assert_scrap_size(scraps, benchmark)

    if k:
        scraps = scraps[:, :k]

    levels = benchmark.dsm.levels[target_index]
    relevance = levels[scraps].astype(float)
    relevance[relevance <= 0] = np.inf
    relevance -= 1
    relevance = np.exp2(-relevance)

    scores = _ndcg(relevance)

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


def _ndcg(ranking: np.ndarray) -> np.ndarray:

    gain = _dcg(ranking)

    y_true_sorted = np.sort(ranking, axis=1)
    y_true_sorted = np.flip(y_true_sorted, axis=1)
    normalizing_gain = _dcg(y_true_sorted)

    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]

    return gain


def _dcg(scores: np.ndarray) -> np.ndarray:

    graded_relevance = np.exp2(scores) - 1

    idx = np.arange(scores.shape[1]) + 1
    reduction = np.log2(idx + 1)
    dcg = graded_relevance / reduction

    return np.sum(dcg, axis=1)
