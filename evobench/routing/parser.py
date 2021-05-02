from pathlib import Path
from typing import Dict, List

from .config import Config, Node


def load(instance_path: Path, solution_path: Path) -> Config:

    capacity: float
    nodes: Dict[Node]
    depot: Node

    global_opt: float
    best_route: List[List[Node]]

    with instance_path.open() as file:
        lines = file.read().split('\n')
        lines = [line.strip() for line in lines]

        capacity_line = next(line for line in lines if line.startswith("CAPACITY"))
        capacity = float(capacity_line.split(":")[1].strip())

        nodes = _parse_nodes(lines)
        depot = next(node for node in nodes.values() if node.is_depot)

    with solution_path.open() as file:
        lines = file.read().split('\n')
        lines = [line.strip() for line in lines]

        global_opt_line = next(line for line in lines if line.startswith("Cost"))
        global_opt = float(global_opt_line.split(" ")[1].strip())
        best_route = _parse_solution(lines, nodes, depot)

    config = Config(
        capacity=capacity,
        nodes=nodes,
        depot=depot,
        global_opt=global_opt,
        best_route=best_route
    )

    return config


def _parse_nodes(lines: List[str]) -> Dict[int, Node]:
    nodes_line = lines.index("NODE_COORD_SECTION")
    demand_line = lines.index("DEMAND_SECTION")
    depot_line = lines.index("DEPOT_SECTION")

    demands = {}

    for i in range(demand_line + 1, depot_line):
        node_id, demand = lines[i].split(" ")
        demands[int(node_id)] = float(demand)

    depot_id = int(lines[depot_line + 1])

    nodes = {}

    for i in range(nodes_line + 1, demand_line):
        node_id, x, y = lines[i].split(" ")
        node_id = int(node_id)
        node = Node(
            id=node_id,
            x=float(x),
            y=float(y),
            demand=demands[node_id],
            is_depot=node_id == depot_id,
        )

        nodes[node_id] = node

    return nodes


def _parse_solution(
    lines: List[str],
    nodes: Dict[int, Node],
    depot: Node
) -> List[Node]:

    lines = [line for line in lines if line.startswith("Route")]

    solution = [depot]

    for line in lines:
        _, node_idx = line.split(":")
        node_idx = node_idx.strip().split(" ")
        route = [nodes[int(node_id)] for node_id in node_idx]

        solution += route
        solution.append(depot)

    solution.append(depot)

    return solution
