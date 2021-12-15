import numpy as np
from lazy import lazy

from .cec2013lsgo import CEC2013LSGO


class F12(CEC2013LSGO):
    """
    7-nonseparable, 1-separable Shifted and Rotated Elliptic Function
    """

    def __init__(
        self,
        *,
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(F12, self).__init__(
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose,
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

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        out_of_bounds = self.check_bounds(x)
        out_of_bounds = np.any(out_of_bounds, axis=1)

        x = x - self.xopt
        fitness = self._rosenbrock(x)

        fitness[out_of_bounds] = None
        return fitness
