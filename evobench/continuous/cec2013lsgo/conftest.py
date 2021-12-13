import numpy as np
from pytest import fixture

from evobench.linkage.dsm import DependencyStructureMatrix
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
    def test_dsm(benchmark: CEC2013LSGO):
        D = benchmark.genome_size
        dsm = benchmark.dsm
        assert isinstance(dsm, DependencyStructureMatrix)
        assert dsm.interactions.shape == (D, D)


@fixture(scope="session")
def helpers() -> Helpers:
    return Helpers
