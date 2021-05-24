from pathlib import Path
from typing import List

import numpy as np
from lazy import lazy
from scipy.spatial.distance import cdist

from evobench.benchmark import Benchmark
from evobench.model.solution import Solution
from evobench.util import shuffle

from .config import Config, Node
from .parser import load


class CVRP(Benchmark):

    def __init__(
        self,
        *,
        suite: str,
        instance: str,
        random_state: int = 42,
        use_shuffle: bool = False,
        multiprocessing: bool = False,
        verbose: int = 0
    ):
        """
        Instantiates _cVRP_ benchmark.
        Instances and benchmark suites were donwloaded from _CVRPLIB_ repository.

        Parameters
        ----------
        suite : str
            Name of the benchmark suite i.e. _A_, _B_, _E_, etc.
        instance : str
            Name of the instance.
        """

        super(CVRP, self).__init__(
            random_state=random_state,
            use_shuffle=use_shuffle,
            multiprocessing=multiprocessing,
            verbose=verbose
        )

        self.suite = suite
        self.instance = instance

    @lazy
    def config(self) -> Config:
        path = Path(__file__).parent
        path = path.joinpath("data").joinpath(self.suite)

        instance_path = path.joinpath(self.instance + ".vrp")
        solution_path = path.joinpath(self.instance + ".sol")

        return load(instance_path, solution_path)

    @lazy
    def genome_size(self) -> int:
        n = len(self.config.target_nodes)
        return n + n - 1

    def random_solutions(self, population_size: int) -> List[Solution]:
        target_nodes_idx = [node.id for node in self.config.target_nodes]
        max_id = max(target_nodes_idx)
        n = len(target_nodes_idx)
        depots = range(max_id + 1, max_id + self.genome_size - n + 1)

        nodes_idx = target_nodes_idx + list(depots)

        genomes = []
        for _ in range(population_size):
            genome = self.random_state.permutation(nodes_idx)
            genomes.append(genome)

        genomes = np.array(genomes)

        if self.USE_SHUFFLE:
            genomes = shuffle(genomes, self.gene_order)

        return list(Solution(genome) for genome in genomes)

    @lazy
    def distances(self) -> np.ndarray:
        coordinates = [(node.x, node.y) for node in self.config.nodes.values()]
        distances = cdist(coordinates, coordinates)
        return distances

    def get_phenome(self, solution: Solution) -> List[Node]:

        depot = self.config.depot
        nodes = self.config.nodes
        genome = list(solution.genome)

        phenome = [
            nodes.get(node_id, depot)
            for node_id in genome
        ]

        phenome = [depot, *phenome, depot]
        fixed_phenome = [depot]
        cargo = 0

        for i in range(1, len(phenome)):
            prev_node = fixed_phenome[-1]
            node = phenome[i]

            # ? removing adjacent depots
            if prev_node.is_depot and node.is_depot:
                continue

            # ? fixing constraints
            if node.is_depot:
                cargo = 0
            elif cargo + node.demand > self.config.capacity:
                fixed_phenome.append(self.config.depot)
                cargo = 0

            fixed_phenome.append(node)
            cargo += node.demand

        return fixed_phenome

    def get_sub_routes(self, phenome: List[Node]) -> List[List[Node]]:
        sub_routes = []
        sub_route = []

        for node in phenome:

            if node.is_depot:
                if sub_route:
                    sub_routes.append(sub_route)

                sub_route = []
            else:
                sub_route.append(node)

        return sub_routes

    def print_route(self, solution: Solution):
        phenome = self.get_phenome(solution)
        sub_routes = self.get_sub_routes(phenome)

        for index, route in enumerate(sub_routes):

            sub_route = [self.config.depot, *route, self.config.depot]
            sub_route_genome = [node.id for node in sub_route]
            sub_route_genome = np.array(sub_route_genome)
            sub_route_solution = Solution(sub_route_genome)

            length = self._evaluate_solution(sub_route_solution)
            cargo = sum([node.demand for node in sub_route])

            print(f"Route {index}")
            print(f"Length: {length:.0f}, Cargo: {cargo}")
            print(f"Nodes: {[node.id for node in route]}\n")

    def are_constraints_satisfied(self, route: List[Node]) -> bool:
        cargo = 0

        for node in route:
            cargo += node.demand

            if node.is_depot:
                cargo = 0

            if cargo > self.config.capacity:
                return False

        return True

    def _evaluate_solution(self, solution: Solution) -> float:
        phenome = self.get_phenome(solution)

        assert self.are_constraints_satisfied(phenome)

        total_distance = 0

        for i, node in enumerate(phenome[1:]):
            prev_node = phenome[i]
            total_distance += self.distances[prev_node.id - 1, node.id - 1]

        return total_distance
