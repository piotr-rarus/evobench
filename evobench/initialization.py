from abc import ABC, abstractmethod
from typing import Dict

from lazy import lazy
from tqdm import tqdm
import numpy as np
from evobench.model import Population


class Initialization(ABC):

    """
    Base class for population initialization.
    """

    def __init__(self, population_size: int, random_seed: int = 0):
        super(Initialization, self).__init__()

        self.POPULATION_SIZE = int(population_size)
        self.RANDOM_SEED = random_seed
        np.random.seed(random_seed)

    def initialize_population(self, genome_size: int) -> Population:
        tqdm.write('\n')
        tqdm.write(
            'Initializing population of {} solutions'
            .format(self.POPULATION_SIZE)
        )
        tqdm.write('\n')

        return self._initialize_population(genome_size)

    @abstractmethod
    def _initialize_population(self, genome_size: int) -> Population:
        pass

    @lazy
    def as_dict(self) -> Dict:
        """
        Initialization description in dictionary format.
        You can dump it as `json` file to log your research.
        """

        as_dict = {}

        as_dict['name'] = self.__class__.__name__
        as_dict['population_size'] = self.POPULATION_SIZE

        return as_dict
