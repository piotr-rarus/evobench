from abc import ABC, abstractmethod
from typing import Dict

from lazy import lazy

from evobench.model import Population


class Initialization(ABC):

    def __init__(self, population_size: int):
        super(Initialization, self).__init__()

        self.POPULATION_SIZE = int(population_size)

    @abstractmethod
    def initialize_population(self, genome_size: int) -> Population:
        pass

    @lazy
    def as_dict(self) -> Dict:
        as_dict = {}

        as_dict['name'] = self.__class__.__name__
        as_dict['population_size'] = self.POPULATION_SIZE

        return as_dict
