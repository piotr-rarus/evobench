from numpy.random import uniform
from tqdm import tqdm

from evobench.initialization import Initialization
from evobench.model import Population, Solution


class Uniform(Initialization):

    def __init__(
        self,
        population_size: int,
        low: float = 0,
        high: float = 1,
        random_seed: int = 0
    ):
        super(Uniform, self).__init__(population_size, random_seed)

        self.low = low
        self.high = high

    def _initialize_population(self, genome_size: int) -> Population:

        solutions = []

        for i in tqdm(range(self.POPULATION_SIZE)):
            genome = uniform(self.low, self.high, size=genome_size)

            solution = Solution(genome)
            solutions.append(solution)

        population = Population(solutions)

        return population
