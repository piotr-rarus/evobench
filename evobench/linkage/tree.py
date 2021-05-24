"""
Helper class for the DSM.
"""

from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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

    def get_fig(self) -> go.Figure:

        pos = self.hierarchy_pos()

        edge_x = []
        edge_y = []

        for edge in self.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        for node in self.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                size=10,
                line_width=2)
            )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                # margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                title="Dependencies"
            ),
        )

        return fig

    def get_levels_fig(self) -> go.Figure:
        levels = self.get_levels()
        levels = pd.DataFrame(
            zip(levels.keys(), levels.values()),
            columns=["Level", "Nodes"]
        )

        levels = levels.sort_values(by="Level")
        fig = px.histogram(
            levels,
            x="Level",
            title="Levels distribution"
        )

        return fig

    def hierarchy_pos(
        self,
        width: float = 1.0,
        vert_gap: float = 0.2,
        vert_loc: float = 0,
        xcenter: float = 0.5
    ):
        """
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
        Licensed under Creative Commons Attribution-Share Alike

        If the graph is a tree this will return the positions to plot this in a
        hierarchical layout.

        G: the graph (must be a tree)

        root: the root node of current branch
        - if the tree is directed and this is not given,
        the root will be found and used
        - if the tree is directed and this is given, then
        the positions will be just for the descendants of this node.
        - if the tree is undirected and not given,
        then a random choice will be used.

        width: horizontal space allocated for this branch - avoids overlap
        with other branches

        vert_gap: gap between levels of hierarchy

        vert_loc: vertical location of root

        xcenter: horizontal location of root
        """

        return self._hierarchy_pos(self.get_root(), width, vert_gap, vert_loc, xcenter)

    def _hierarchy_pos(
        self, root: int,
        width: float = 1.0, vert_gap: float = 0.2, vert_loc: float = 0,
        xcenter: float = 0.5, pos=None, parent=None
    ) -> Dict[int, Tuple[float, float]]:
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(self.neighbors(root))

        if len(children) != 0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = self._hierarchy_pos(
                    child, width=dx, vert_gap=vert_gap,
                    vert_loc=vert_loc-vert_gap, xcenter=nextx,
                    pos=pos, parent=root
                )

        return pos
