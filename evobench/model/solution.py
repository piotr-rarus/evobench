from dataclasses import dataclass
from typing import List, Optional, Union

from numpy import ndarray


@dataclass
class Solution:
    genome: ndarray
    fitness: Optional[Union[float, List[float]]] = None

    def __hash__(self) -> int:
        return hash(bytes(self.genome))
