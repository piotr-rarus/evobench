from dataclasses import dataclass
from typing import Dict, List

from lazy import lazy


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    demand: float
    is_depot: bool


@dataclass(frozen=True)
class Config:
    capacity: float
    nodes: Dict[int, Node]
    depot: Node
    global_opt: float
    best_route: List[Node]

    @lazy
    def target_nodes(self) -> List[Node]:
        return [node for node in self.nodes.values() if not node.is_depot]
