# from typing import List

# import numpy as np
# from lazy import lazy

# from evobench.separable import Separable


# class MaxSat(Separable):

#     def __init__(
#         self,
#         k_clauses: int,
#         block_size: int,
#         repetitions: int,
#         overlap_size: int = 0,
#         random_seed: int = 0
#     ):
#         super(MaxSat, self).__init__(block_size, repetitions, overlap_size)

#         self.K_CLAUSES = k_clauses
#         self.GLOBAL_OPTIMUM = k_clauses * repetitions
#         self.RANDOM_SEED = random_seed

#     @lazy
#     def clauses(self) -> List[List[np.ndarray]]:
#         np.random.seed(self.RANDOM_SEED)
#         clauses = []

#         for n in range(self.REPETITIONS):
#             block_clauses = []

#             for k in range(self.K_CLAUSES):
#                 clause = np.random.randint(low=0, high=2, size=self.BLOCK_SIZE)
#                 block_clauses.append(clause)

#             clauses.append(block_clauses)

#         return clauses

#     def evaluate_block(self, block: np.ndarray, block_index: int) -> int:
#         total = 0
#         block_clauses = self.clauses[block_index]

#         for clause in block_clauses:
#             same = block == clause
#             if np.sum(same):
#                 total += 1

#         return total
