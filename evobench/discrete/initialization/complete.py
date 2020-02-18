from random import shuffle

import numpy as np
from sympy.utilities.iterables import variations

from evobench.initialization import Initialization
from evobench.model import Population, Solution


class Complete(Initialization):
    """
    Returns all samples from a given discrete distribution.
    Sometimes useful ;)
    """

    def __init__(self, population_size: int, random_seed: int = 0):
        super().__init__(population_size)

    def _initialize_population(self, genome_size: int) -> Population:

        solutions = variations([0, 1], genome_size, repetition=True)

        solutions = [Solution(np.array(solution)) for solution in solutions]
        shuffle(solutions)
        solutions = list(solutions)

        return Population(solutions)
