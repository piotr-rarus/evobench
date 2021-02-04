from typing import List

import numpy as np
from evobench.discrete.trap import Trap

from ..fill_quality import get_fill_quality_from_dsm


def test_dsm_fill_quality():
    benchmark = Trap(blocks=[2, 1, 3])

    pred_dsm = [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1]
    ]

    pred_dsm = np.array(pred_dsm)

    fill_quality = get_fill_quality_from_dsm(pred_dsm, benchmark.true_dsm)

    assert isinstance(fill_quality, List)
    assert len(fill_quality) == 5

    assert fill_quality == [1, 1, 0, 0.5, 0.5]
