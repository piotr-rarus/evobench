from abc import abstractmethod
from pathlib import Path

import numpy as np
from lazy import lazy

from evobench.continuous.continuous import Continuous
from evobench.model import Population


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

    @lazy
    def _data_path(self) -> Path:
        path = Path(__file__).parent
        path = path.joinpath("data")
        return path

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
        y = self._evaluate_population(x)

        for solution, fitness in zip(population.get_not_evaluated_solutions(), y):
            solution.fitness = fitness

        return population.fitness

    @abstractmethod
    def _evaluate_population(self, x: np.ndarray) -> np.ndarray:
        pass

    def _elliptic(self, x: np.ndarray) -> np.ndarray:
        D = self.genome_size
        condition = 1e+6
        coefficients = condition ** np.linspace(0, 1, D)
        fit = coefficients @ (self._T_irreg(x) ** 2).T
        return fit

    def _T_irreg(self, x: np.ndarray) -> np.ndarray:
        """
        This transformation is used to create smooth local irregularities.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray
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

# %------------------------------------------------------------------------------
# % Rastrigin's Function
# %------------------------------------------------------------------------------
# function fit = rastrigin(x)
#     [D ps] = size(x);
#     A = 10;
#     x = T_diag(T_asy(T_irreg(x), 0.2), 10);
#     fit = A*(D - sum(cos(2*pi*x), 1)) + sum(x.^2, 1);
# end


#   [5, 1000] = [5, 1]

# [1, 1000]  [1000, 5] = [5, 1]


# end

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %------------------------------------------------------------------------------
# % This transformation function is used to break the symmetry of symmetric
# % functions.
# %------------------------------------------------------------------------------
# function g = T_asy(f, beta)
#     [D popsize] = size(f);
#     g = f;
#     temp = repmat(beta * linspace(0, 1, D)', 1, popsize);
#     ind = f > 0;
#     g(ind) = f(ind).^ (1 + temp(ind) .* sqrt(f(ind)));
# end


# %------------------------------------------------------------------------------
# % This transformation is used to create the ill-conditioning effect.
# %------------------------------------------------------------------------------
# function g = T_diag(f, alpha)
#     [D popsize] = size(f);
#     scales = repmat(sqrt(alpha) .^ linspace(0, 1, D)', 1, popsize);
#     g = scales .* f;
# end


# %------------------------------------------------------------------------------
# % This transformation is used to create smooth local irregularities.
# %------------------------------------------------------------------------------
# function g = T_irreg(f)
#    a = 0.1;
#    g = f;
#    idx = (f > 0);
#    g(idx) = log(f(idx))/a;
#    g(idx) = exp(g(idx) + 0.49*(sin(g(idx)) + sin(0.79*g(idx)))).^a;
#    idx = (f < 0);
#    g(idx) = log(-f(idx))/a;
#    g(idx) = -exp(g(idx) + 0.49*(sin(0.55*g(idx)) + sin(0.31*g(idx)))).^a;
# end

# %------------------------------------------------------------------------------
# % This function tests a given decision vector against the boundaries of a function.
# %------------------------------------------------------------------------------
# function indices = checkBounds(x, lb, ub)
#     indices = find(sum(x > ub | x < lb) > 0);
# end

# %------------------------------------------------------------------------------
# % Elliptic Function
# %------------------------------------------------------------------------------
# function fit = elliptic(x)
#     %TODO Do we need symmetry breaking?
#     %TODO Implement to support a matrix as input.

#     [D ps] = size(x);
#     condition = 1e+6;
#     coefficients = condition .^ linspace(0, 1, D);
#     fit = coefficients * T_irreg(x).^2;
# end

# %------------------------------------------------------------------------------
# % Ackley's Function
# %------------------------------------------------------------------------------
# function fit = ackley(x)
#     [D ps] = size(x);
#     x = T_irreg(x);
#     x = T_asy(x, 0.2);
#     x = T_diag(x, 10);
#     fit = sum(x.^2,1);
#     fit = 20-20.*exp(-0.2.*sqrt(fit./D))-exp(sum(cos(2.*pi.*x),1)./D)+exp(1);
# end


# %------------------------------------------------------------------------------
# % Schwefel's Problem 1.2
# %------------------------------------------------------------------------------
# function fit = schwefel(x)
#     [D ps] = size(x);
#     x = T_asy(T_irreg(x), 0.2);
#     fit = 0;
#     for i = 1:D
#         fit = fit + sum(x(1:i,:),1).^2;
#     end
# end


# %------------------------------------------------------------------------------
# % Rosenbrock's Function
# %------------------------------------------------------------------------------
# function fit = rosenbrock(x)
#     [D ps] = size(x);
#     fit = sum(100.*(x(1:D-1,:).^2-x(2:D, :)).^2+(x(1:D-1, :)-1).^2);
# end
