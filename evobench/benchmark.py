import warnings
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List

import numpy as np
from lazy import lazy
from tqdm.auto import tqdm

from evobench.model.population import Population
from evobench.model.solution import Solution
from evobench.util import deshuffle_solution


class Benchmark(ABC):
    """
    Base class for problem encapsulation.
    If you wish to implement your own problem, please
    inherit from this class.
    """

    def __init__(
        self,
        rng_seed: int = 42,
        use_shuffle: bool = False,
        verbose: int = 0
    ):
        super(Benchmark, self).__init__()
        self.RNG_SEED = rng_seed
        self.USE_SHUFFLE = use_shuffle
        self.VERBOSE = verbose

        self.ffe = 0
        self.rng = np.random.default_rng(rng_seed)

    @abstractproperty
    def genome_size(self) -> int:
        pass

    @lazy
    def gene_order(self) -> np.ndarray:
        gene_order = np.arange(self.genome_size)

        if self.USE_SHUFFLE:
            self.rng.shuffle(gene_order)

        return gene_order

    @lazy
    def lower_bound(self) -> np.ndarray:
        pass

    @lazy
    def upper_bound(self) -> np.ndarray:
        pass

    @lazy
    def bound_range(self) -> np.ndarray:
        return self.upper_bound - self.lower_bound

    @lazy
    def as_dict(self) -> Dict:
        """
        Benchmark description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}

        as_dict["name"] = self.__class__.__name__
        as_dict["genome_size"] = self.genome_size
        as_dict["shuffle"] = self.USE_SHUFFLE

        return as_dict

    def random_solutions(self, population_size: int) -> List[Solution]:
        pass

    def initialize_population(self, population_size: int) -> Population:
        solutions = []
        population_size = int(population_size)

        if self.VERBOSE:
            print(f"\nInitializing poptulation: {population_size} samples")

        solutions = self.random_solutions(population_size)
        return Population(solutions)

    def fix(self, solution: Solution) -> Solution:
        genome = solution.genome.copy()

        mask = genome > self.upper_bound
        genome[mask] = self.upper_bound[mask]

        mask = genome < self.lower_bound
        genome[mask] = self.lower_bound[mask]

        return Solution(genome)

    def check_bounds(self, x: np.ndarray) -> np.ndarray:
        mask = x > self.upper_bound
        mask += x < self.lower_bound
        return mask

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

        solutions = population.get_not_evaluated_solutions()

        if self.VERBOSE:
            print(f"\nEvaluating population of {population.size} solutions\n")
            solutions = tqdm(solutions)

        for solution in solutions:
            solution.fitness = self.evaluate_solution(solution)

        return population.fitness

    def evaluate_solution(self, solution: Solution) -> float:
        """
        Evaluate fitness of a single solution.

        Parameters
        ----------
        solution : Solution
            Genome wrapped as `Solution`.

        Returns
        -------
        float
            Fitness value.
        """

        assert solution.genome.size == self.genome_size
        self.ffe += 1

        if self.USE_SHUFFLE:
            solution = deshuffle_solution(solution, self.gene_order)

        bounds_violated = np.any(self.check_bounds(solution.genome))
        if bounds_violated:
            warnings.warn(
                f"Solution {solution.__hash__} is violating boundary constraints."
            )
            return None

        return self._evaluate_solution(solution)

    @abstractmethod
    def _evaluate_solution(self, solution: Solution) -> float:
        pass
