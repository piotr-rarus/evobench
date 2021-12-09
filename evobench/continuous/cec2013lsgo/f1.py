import numpy as np
from lazy import lazy

from evobench.model import Solution

from .cec2013lsgo import CEC2013LSGO


class F1(CEC2013LSGO):

    def __init__(
        self,
        *,
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(F1, self).__init__(
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose
        )

    @property
    def genome_size(self) -> np.ndarray:
        return 1_000

    @lazy
    def lower_bound(self) -> np.ndarray:
        lower_bound = [-100] * self.genome_size
        return np.array(lower_bound)

    @lazy
    def upper_bound(self) -> np.ndarray:
        upper_bound = [100] * self.genome_size
        return np.array(upper_bound)

    @lazy
    def xopt(self) -> np.ndarray:
        path = self._data_path.joinpath("F1-xopt.txt")
        xopt = np.loadtxt(path)
        return xopt

    def _evaluate_solution(self, solution: Solution) -> float:
        x = solution.genome - self.xopt
        fitness = self._elliptic(x)
        return fitness

    def _evaluate_population(self, x: np.ndarray) -> np.ndarray:
        out_of_bounds = self.check_bounds(x)
        out_of_bounds = np.any(out_of_bounds, axis=1)
        x -= self.xopt
        fitness = self._elliptic(x)
        fitness[out_of_bounds] = None
        return fitness
