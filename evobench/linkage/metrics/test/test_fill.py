# import numpy as np
# from evobench.discrete.trap import Trap

# from ..fill import fill_quality


# def test_dsm_fill_quality():
#     benchmark = Trap(blocks=[2, 1, 3])

#     pred_dsm = [
#         [1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1, 1],
#         [0, 0, 0, 0, 1, 1]
#     ]

#     pred_dsm = np.array(pred_dsm)

#     scores = fill_quality(pred_dsm, benchmark.true_dsm)

#     assert isinstance(scores, list)
#     assert len(scores) == 5

#     assert scores == [1, 1, 0, 0.5, 0.5]
