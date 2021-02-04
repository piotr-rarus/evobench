from typing import List

import numpy as np
from evobench.linkage.dsm import get_block_width, get_ils
from tqdm.auto import tqdm


def get_fill_quality_from_dsm(
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
        block_width = get_block_width(gene_index, true_dsm)
        true_ils = get_ils(gene_index, true_dsm, block_width)

        if not true_ils:
            continue

        pred_ils = get_ils(gene_index, pred_dsm, block_width)

        pred_ils = set(pred_ils)
        true_ils = set(true_ils)

        pred_positive = true_ils.intersection(pred_ils)

        quality = len(pred_positive) / len(true_ils)
        fill_quality.append(quality)

    return fill_quality
