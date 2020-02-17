from numpy.random import uniform

from evobench.initialization import Initialization
from evobench.model import Population, Solution


class Uniform(Initialization):

    def __init__(self, population_size: int, low: float = 0, high: float = 1):
        super(Uniform, self).__init__(population_size)

        self.low = low
        self.high = high

    def initialize_population(self, genome_size: int) -> Population:

        solutions = []

        for i in range(self.POPULATION_SIZE):
            genome = uniform(self.low, self.high, size=genome_size)

            solution = Solution(genome)
            solutions.append(solution)

        solutions = tuple(solutions)
        population = Population(solutions)

        return population
