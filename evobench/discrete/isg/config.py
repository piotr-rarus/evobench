from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from lazy import lazy

from .spin import Spin


@dataclass(frozen=True)
class Config:
    name: str
    min_energy: int
    best_solution: np.ndarray
    spins: List[Spin]

    @lazy
    def genome_size(self) -> int:
        return self.best_solution.size

    @lazy
    def span(self) -> int:
        return len(self.spins) - self.min_energy

    @lazy
    def a_spin_indices(self) -> np.ndarray:
        return np.array([spin.a_index for spin in self.spins])

    @lazy
    def b_spin_indices(self) -> np.ndarray:
        return np.array([spin.b_index for spin in self.spins])

    @lazy
    def spin_factors(self) -> np.ndarray:
        return np.array([spin.factor for spin in self.spins])

    @lazy
    def as_dict(self) -> Dict:
        as_dict = {}
        as_dict['name'] = self.name
        as_dict['min_energy'] = self.min_energy
        as_dict['best_solution'] = list(self.best_solution)
        as_dict['configs_len'] = len(self.spins)

        return as_dict
