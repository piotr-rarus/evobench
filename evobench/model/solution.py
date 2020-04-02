from dataclasses import dataclass
from lazy import lazy
from numpy import ndarray
from hashlib import blake2b


@dataclass(frozen=True)
class Solution:
    genome: ndarray

    @lazy
    def __hash__(self):
        return blake2b(self.genome.tobytes()).hexdigest()
