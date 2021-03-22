from pathlib import Path
from typing import Dict

import numpy as np
from evobench.discrete import Discrete
from evobench.model import Solution
from lazy import lazy

from .config import Config
from .parser import load


class IsingSpinGlass(Discrete):

    def __init__(
        self,
        config_name: str,
        *,
        random_state: int = 42,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        """
        Instantiates _ISG_ benchmark

        Parameters
        ----------
        config_name : str
            Name of configuration file, without suffix.
            Predefined configurations can be found at
            `evobench.discrete.isg.data`. These problem files are ported from
            _P3_ repository.
        """

        super(IsingSpinGlass, self).__init__(
            random_state=random_state,
            use_shuffle=use_shuffle,
            multiprocessing=multiprocessing,
            verbose=verbose,
        )

        self.config_name = config_name

    @lazy
    def config(self) -> Config:
        path = Path(__file__).parent
        path = path.joinpath('data')
        path = path.joinpath(self.config_name + '.txt')

        return load(path)

    @lazy
    def true_dsm(self) -> np.ndarray:
        dsm = np.eye(self.config.genome_size)

        for spin in self.config.spins:

            dsm[spin.a_index, spin.b_index] = 1
            dsm[spin.b_index, spin.a_index] = 1

        return dsm

    @lazy
    def genome_size(self) -> int:
        return self.config.genome_size

    @lazy
    def as_dict(self) -> Dict:
        as_dict = self.config.as_dict

        benchmark_as_dict = super().as_dict
        as_dict = {**benchmark_as_dict, **as_dict}

        return as_dict

    def _evaluate_solution(self, solution: Solution) -> float:

        genome = solution.genome.copy()
        genome[solution.genome == 0] = -1

        a_genes = genome[self.config.a_spin_indices]
        b_genes = genome[self.config.b_spin_indices]

        spins = a_genes * b_genes * self.config.spin_factors

        energy = - spins.sum()

        fitness = (energy - self.config.min_energy) / self.config.span
        return 1 - fitness
