from dataclasses import dataclass

from numpy import ndarray


@dataclass(frozen=True)
class Solution:
    genome: ndarray
