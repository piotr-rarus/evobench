import numpy as np
from lazy import lazy

from .cec2013lsgo import CEC2013LSGO


class F14(CEC2013LSGO):
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
        super(F14, self).__init__(
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose,
        )

        self.c = np.cumsum(self.s)
        self.m = 5

    @property
    def genome_size(self) -> np.ndarray:
        return 905

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

        fitness = 0
        ldim = 0
        ldimshift = 0

        for i in range(len(self.s)):
            if i > 0:
                ldim = self.c[i-1] - i * self.m
                ldimshift = self.c[i-1]

            udim = self.c[i] - i * self.m
            udimshift = self.c[i]

            f: np.ndarray
            xopt_shift = self.xopt[ldimshift:udimshift].reshape(1, -1)
            z = x[:, self.p[ldim:udim] - 1] - xopt_shift

            if self.s[i] == 25:
                f = self.R25
            elif self.s[i] == 50:
                f = self.R50
            elif self.s[i] == 100:
                f = self.R100

            f = f @ z.T
            f = self._schwefel(f.T)
            fitness += self.w[i] * f

        fitness[out_of_bounds] = None
        return fitness
