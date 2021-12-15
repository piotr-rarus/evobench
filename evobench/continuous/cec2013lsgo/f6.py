import numpy as np
from lazy import lazy

from .cec2013lsgo import CEC2013LSGO


class F6(CEC2013LSGO):
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
        super(F6, self).__init__(
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose,
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

        fitness = 0
        ldim = 0

        for i in range(len(self.s)):
            f: np.ndarray
            z = x[:, self.p[ldim:ldim + self.s[i]] - 1].T
            ldim += self.s[i]

            if self.s[i] == 25:
                f = self.R25
            elif self.s[i] == 50:
                f = self.R50
            elif self.s[i] == 100:
                f = self.R100

            f = f @ z
            f = self._ackley(f.T)
            fitness += self.w[i] * f

        fitness += self._ackley(x[:, self.p[ldim:] - 1])

        fitness[out_of_bounds] = None
        return fitness
