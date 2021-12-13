from abc import abstractmethod
from pathlib import Path

import numpy as np
from lazy import lazy

from evobench.continuous.continuous import Continuous
from evobench.model import Population, Solution


class CEC2013LSGO(Continuous):

    def __init__(
        self,
        *,
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(CEC2013LSGO, self).__init__(
            rng_seed=rng_seed,
            use_shuffle=use_shuffle,
            verbose=verbose
        )

        self._load_data_files()

    @lazy
    def _data_path(self) -> Path:
        path = Path(__file__).parent
        path = path.joinpath("data")
        return path

    def _load_data_files(self):
        fn = self.__class__.__name__
        for data_file in self._data_path.glob(f"{fn}-*.txt"):
            data = np.loadtxt(data_file, delimiter=",")
            attr_name = data_file.stem.replace(f"{fn}-", "")
            if attr_name in ["p", "s"]:
                data = data.astype(int)
            setattr(self, attr_name, data)

    def evaluate_population(self, population: Population) -> np.ndarray:
        """
        Evaluates population of solutions.

        Parameters
        ----------
        population : Population
            Collection of solutions wrapped as `Population`.

        Returns
        -------
        np.ndarray
            An array of fitness values.
            Order is the same as input population.
        """

        x = population.as_ndarray[~population.evaluated_mask]
        y = self._evaluate(x)

        for solution, fitness in zip(population.get_not_evaluated_solutions(), y):
            solution.fitness = fitness

        return population.fitness

    def _evaluate_solution(self, solution: Solution) -> float:
        x = solution.genome.reshape(1, -1)
        fitness = self._evaluate(x)
        return fitness[0]

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    def _sphere(self, x: np.ndarray) -> np.ndarray:
        fit = np.sum(x ** 2, axis=-1)
        return fit

    def _elliptic(self, x: np.ndarray) -> np.ndarray:
        D = x.shape[-1]
        condition = 1e+6
        coefficients = condition ** np.linspace(0, 1, D)
        fit = coefficients @ (self._T_irreg(x) ** 2).T
        return fit

    def _rastrigin(self, x: np.ndarray) -> np.ndarray:
        D = x.shape[-1]
        A = 10
        x = self._T_irreg(x)
        x = self._T_asy(x, beta=0.2)
        x = self._T_diag(x, alpha=10)
        fit = A * (D - np.sum(np.cos(2 * np.pi * x), axis=-1)) + np.sum(x ** 2, axis=-1)
        return fit

    def _ackley(self, x: np.ndarray) -> np.ndarray:
        D = x.shape[-1]
        x = self._T_irreg(x)
        x = self._T_asy(x, beta=0.2)
        x = self._T_diag(x, alpha=10)
        fit = np.sum(x ** 2, axis=-1)
        fit = 20 - 20 * np.exp(-0.2 * np.sqrt(fit / D))
        fit -= np.exp(np.sum(np.cos(2 * np.pi * x), axis=-1) / D)
        fit += np.exp(1)
        return fit

    def _schwefel(self, x: np.ndarray) -> np.ndarray:
        D = x.shape[-1]
        x = self._T_irreg(x)
        x = self._T_asy(x, beta=0.2)
        fit = 0
        for i in range(D):
            fit += np.sum(x[:, :i+1], axis=-1) ** 2

        return fit

    def _rosenbrock(self, x: np.ndarray) -> np.ndarray:
        D = x.shape[-1]
        x = 100 * (x[:, 0:D-1] ** 2 - x[:, 1:D]) ** 2 + (x[:, 0:D-1]-1) ** 2
        fit = np.sum(x, axis=-1)
        return fit

    def _T_asy(self, x: np.ndarray, beta: float) -> np.ndarray:
        """
        This transformation function is used to break the symmetry of symmetric
        functions.
        """
        pop_size, D = x.shape

        g = x.copy()
        temp = beta * np.linspace(0, 1, D).reshape(1, -1)

        if pop_size > 1:
            temp = np.repeat(temp.reshape(1, -1), pop_size, axis=0)

        idx = x > 0
        g[idx] = x[idx] ** (1 + temp[idx] * np.sqrt(x[idx]))
        return g

    def _T_diag(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """
        This transformation is used to create the ill-conditioning effect.
        """
        D = x.shape[-1]
        scales = np.sqrt(alpha) ** np.linspace(0, 1, D)
        g = scales * x
        return g

    def _T_irreg(self, x: np.ndarray) -> np.ndarray:
        """
        This transformation is used to create smooth local irregularities.
        """
        a = 0.1
        idx = x > 0
        g = x.copy()
        g[idx] = np.log(x[idx]) / a
        g[idx] = np.exp(g[idx] + 0.49 * (np.sin(g[idx]) + np.sin(0.79 * g[idx]))) ** a
        idx = x < 0
        g[idx] = np.log(-x[idx]) / a
        g[idx] = -np.exp(g[idx] + 0.49*(np.sin(0.55*g[idx]) + np.sin(0.31*g[idx]))) ** a

        return g
