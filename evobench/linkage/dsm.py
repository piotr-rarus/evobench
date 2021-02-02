import numpy as np
from typing import List


def get_block_width(
    gene_index: int,
    true_dsm: np.ndarray
) -> int:
    genome_size, _ = true_dsm.shape
    interactions = true_dsm[gene_index, :]
    positive = interactions == 1
    return positive.sum()


def get_ils(
    gene_index: int,
    dsm: np.ndarray,
    block_width: int = None
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
        By default, calculate ILS for whole genome.

    Returns
    -------
    List[int]
        Incremental Linkage Set
    """

    ils = []
    ils.append(gene_index)
    genome_size, _ = dsm.shape

    if block_width is None:
        block_width = genome_size

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
