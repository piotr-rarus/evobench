import numpy as np
from lazy import lazy

from .cec2013lsgo import CEC2013LSGO


class F3(CEC2013LSGO):

    def __init__(
        self,
        *,
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(F3, self).__init__(
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose
        )

    @property
    def genome_size(self) -> np.ndarray:
        return 1_000

    @lazy
    def lower_bound(self) -> np.ndarray:
        lower_bound = [-32] * self.genome_size
        return np.array(lower_bound)

    @lazy
    def upper_bound(self) -> np.ndarray:
        upper_bound = [32] * self.genome_size
        return np.array(upper_bound)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        out_of_bounds = self.check_bounds(x)
        out_of_bounds = np.any(out_of_bounds, axis=1)
        x = x - self.xopt
        fitness = self._ackley(x)
        fitness[out_of_bounds] = None
        return fitness
