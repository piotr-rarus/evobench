from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from evobench.benchmark import Benchmark
from evobench.model.solution import Solution


def check_samples(
    samples: List[Tuple[np.ndarray, float]],
    benchmark: Benchmark
):
    for genome, fitness in samples:
        solution = Solution(np.array(genome))
        pred_fitness = benchmark.evaluate_solution(solution)

        assert isinstance(pred_fitness, float)
        assert pred_fitness == fitness


def dsm_fill_quality(
    pred_dsm: np.ndarray,
    true_dsm: np.ndarray
) -> List[float]:
    """
    On measuring and improving the quality of linkage learning in
    modern evolutionary algorithms applied to solve partially additively
    separable problems.

    Michal W. Przewozniczek, Bartosz Frej, Marcin M. Komarnicki

    Calculates fill quality linkage metric based on true and predicted DSM.
    Metric was proposed in the paper mentioned above. It's applicable
    only for partially separable problems.
    It won't work for overlapping ones.

    Parameters
    ----------
    pred_dsm : np.ndarray
        Predicted DSM for which linkage quality will be calculated.
    true_dsm : np.ndarray
        Ground truth DSM.

    Returns
    -------
    List[float]
        Distribution of fill quality metric for each gene in the genome.
    """

    tqdm.write('Calculating DSM fill quality')

    fill_quality = []
    genome_size, _ = pred_dsm.shape

    for gene_index in range(genome_size):
        block_width = _get_block_width(gene_index, true_dsm)
        true_ils = _get_ils(gene_index, true_dsm, block_width)

        if not true_ils:
            continue

        pred_ils = _get_ils(gene_index, pred_dsm, block_width)

        pred_ils = set(pred_ils)
        true_ils = set(true_ils)

        pred_positive = true_ils.intersection(pred_ils)

        quality = len(pred_positive) / len(true_ils)
        fill_quality.append(quality)

    return fill_quality


def _get_ils(
    gene_index: int,
    dsm: np.ndarray,
    block_width: int
) -> List[int]:
    """
    Optimization by Pairwise Linkage Detection,
    Incremental Linkage Set, and Restricted / Back Mixing: DSMGA-II

    Shih-Huan Hsu, Tian-Li Yu

    arXiv:1807.11669

    Calculates Incremental Linkage Set for a given gene and DSM.

    Parameters
    ----------
    gene_index : int
        Gene which starts the sequence of dependencies.
    dsm : np.ndarray
        Dependency structure matrix
    block_width : int
        Block width for given index. Used to speed up computation.

    Returns
    -------
    List[int]
        Incremental Linkage Set
    """

    ils = []
    ils.append(gene_index)
    genome_size, _ = dsm.shape

    while len(ils) < block_width:
        last_gene_index = ils[-1]

        available_genes = [
            gene_index
            for gene_index in range(0, genome_size)
            if gene_index not in ils
        ]

        dependencies = [
            dsm[last_gene_index, available_gene]
            for available_gene in available_genes
        ]

        max_index = np.argmax(dependencies)
        ils.append(available_genes[max_index])

    return ils[1:]


def _get_block_width(
    gene_index: int,
    true_dsm: np.ndarray
) -> int:
    block_size = 0
    genome_size, _ = true_dsm.shape

    for i in range(genome_size):
        interaction = true_dsm[gene_index, i]

        if interaction == 1:
            block_size += 1

    return block_size
