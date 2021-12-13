import numpy as np
from pytest import fixture

from evobench.model import Population, Solution

from .cec2013lsgo import CEC2013LSGO


class Helpers:

    @staticmethod
    def test_evaluate_solution(benchmark: CEC2013LSGO):
        genome = np.zeros(shape=benchmark.genome_size)
        solution = Solution(genome)
        fitness = benchmark.evaluate_solution(solution)
        assert isinstance(fitness, float)

    @staticmethod
    def test_evaluate_population(benchmark: CEC2013LSGO):
        genomes = np.ones(shape=(5, benchmark.genome_size))
        factors = np.array([0, 1, 2, 3, 300])
        genomes = genomes * factors[:, None]
        solutions = list(Solution(genome) for genome in genomes)
        population = Population(solutions)
        fitness = benchmark.evaluate_population(population)
        assert isinstance(fitness, np.ndarray)
        assert len(fitness) == population.size

    @staticmethod
    def test_data_files(benchmark: CEC2013LSGO):
        assert isinstance(benchmark.p, np.ndarray)
        assert isinstance(benchmark.R25, np.ndarray)
        assert isinstance(benchmark.R50, np.ndarray)
        assert isinstance(benchmark.R100, np.ndarray)
        assert isinstance(benchmark.s, np.ndarray)
        assert isinstance(benchmark.w, np.ndarray)
        assert isinstance(benchmark.xopt, np.ndarray)

        assert benchmark.p.dtype is int
        assert benchmark.s.dtype is int


@fixture(scope="session")
def helpers() -> Helpers:
    return Helpers
