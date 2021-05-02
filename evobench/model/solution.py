from dataclasses import dataclass
from typing import Optional

from numpy import ndarray


@dataclass
class Solution:
    genome: ndarray
    fitness: Optional[float] = None

    def __hash__(self) -> int:
        return id(self.genome)
