from pytest import fixture

from evobench.initialization import Initialization
from evobench.model import Population

from evobench.discrete.initialization import Uniform

__GENOME_SIZE = 10
__POPULATION_SIZE = 10


@fixture(scope='session')
def initialization() -> Initialization:
    initialization = Uniform(__POPULATION_SIZE)
    return initialization


@fixture(scope='session')
def population(initialization: Initialization) -> Population:
    population = initialization.initialize_population(__GENOME_SIZE)

    return population
