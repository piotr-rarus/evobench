from numpy.random import randint

from evobench.model import Population, Solution
from evobench.initialization import Initialization


class Uniform(Initialization):

    def __init__(self, population_size: int, random_seed: int = 0):
        super().__init__(population_size, random_seed)

    def _initialize_population(self, genome_size: int) -> Population:

        solutions = []

        for i in range(self.POPULATION_SIZE):
            genome = randint(low=0, high=2, size=genome_size)
            solution = Solution(genome)
            solutions.append(solution)

        solutions = tuple(solutions)
        population = Population(solutions)

        return population
