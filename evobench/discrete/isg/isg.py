from pathlib import Path
from typing import Dict

import numpy as np
from lazy import lazy

from evobench.benchmark import Benchmark
from evobench.model import Solution

from .config import Config
from .parser import load


class IsingSpinGlass(Benchmark):

    def __init__(self, config_name: str, shuffle: bool = False):
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

        super(IsingSpinGlass, self).__init__(shuffle)
        self.config_name = config_name

    @lazy
    def config(self) -> Config:
        path = Path(__file__).parent
        path = path.joinpath('data')
        path = path.joinpath(self.config_name + '.txt')

        return load(path)

    @lazy
    def global_opt(self) -> float:
        return 1

    @lazy
    def dsm(self) -> np.ndarray:
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

        energy = 0.0

        genome = solution.genome.copy()
        genome[solution.genome == 0] = -1

        for spin in self.config.spins:
            a_gene = solution.genome[spin.a_index]
            b_gene = solution.genome[spin.b_index]

            spin = a_gene * b_gene * spin.factor
            energy -= spin

        score = (energy - self.config.min_energy) / self.config.span
        return 1 - score
