"""
Helper class for the DSM.
"""

from typing import Dict, List

import networkx as nx


class Tree(nx.DiGraph):

    def __init__(self):
        super(Tree, self).__init__()

    def get_root(self) -> int:
        root = next(
            node
            for node, in_degree
            in self.in_degree()
            if in_degree == 0
        )

        return root

    def get_leaves(self) -> List[int]:

        leaves = [
            node
            for node, out_degree
            in self.out_degree()
            if out_degree == 0
        ]

        return leaves

    def get_levels(self) -> Dict[int, List[int]]:
        root = self.get_root()
        level = 0
        levels = {level: [root]}
        last_level = [root]

        while last_level:
            next_level = []
            level += 1
            level_nodes = []

            for node in last_level:
                children = [child for parent, child in self.out_edges(node)]
                level_nodes += children
                next_level += children

            if next_level:
                levels[level] = next_level

            last_level = next_level

        return levels

    def get_depth(self) -> int:
        levels = self.get_levels()
        return len(levels.keys())

    def get_node_level(self, node: int) -> int:
        for level, nodes in self.levels.items():
            if node in nodes:
                return level
