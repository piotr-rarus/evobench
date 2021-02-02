from dataclasses import dataclass
from hashlib import blake2b
from typing import Optional

from lazy import lazy
from numpy import ndarray


@dataclass()
class Solution:
    genome: ndarray
    fitness: Optional[float] = None

    @lazy
    def __hash__(self):
        return blake2b(self.genome.tobytes(), digest_size=10).hexdigest()
