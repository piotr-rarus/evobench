from typing import List

import numpy as np
import plotly.graph_objects as go
from lazy import lazy

from evobench.linkage.tree import Tree


class DependencyStructureMatrix:

    def __init__(self, interactions: np.ndarray):

        assert interactions.shape[0] == interactions.shape[1]
        assert interactions.min() >= 0
        assert interactions.max() <= 1

        self.interactions = interactions
        self.GENOME_SIZE = interactions.shape[0]

    @lazy
    def ils(self) -> List[np.ndarray]:
        ils = []

        for gene_index in range(self.GENOME_SIZE):
            gene_ils = self._get_ils(gene_index)
            ils.append(gene_ils)

        return ils

    @lazy
    def trees(self) -> List[Tree]:
        trees = []

        for gene_index in range(self.GENOME_SIZE):
            tree = self._get_tree(gene_index)
            trees.append(tree)

        return trees

    @lazy
    def levels(self) -> np.ndarray:
        levels = []

        for gene_index in range(self.GENOME_SIZE):
            gene_levels = np.full(self.GENOME_SIZE, -1)
            gene_tree = self.trees[gene_index]
            gene_tree_levels = gene_tree.get_levels()

            for level, genes in gene_tree_levels.items():
                gene_levels[genes] = level

            levels.append(gene_levels)

        return np.vstack(levels)

    def get_block_width(self, gene_index: int) -> int:
        interactions = self.interactions[gene_index, :]
        positive = interactions == 1
        return positive.sum()

    def get_fig(
        self,
        title: str = "Dependency Structure Matrix",
        colorscale: str = "Brwnyl_r"
    ) -> go.Figure:

        fig = go.Figure()
        heatmap = go.Heatmap(
            z=self.interactions,
            colorscale=colorscale,
        )
        fig.add_trace(heatmap)
        fig.update_layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(scaleanchor="x", autorange="reversed", visible=False),
        )

        return fig

    def _get_ils(self, gene_index: int, block_width: int = None) -> np.ndarray:
        """
        Calculates Incremental Linkage Set for a given gene and DSM.

        Optimization by Pairwise Linkage Detection,
        Incremental Linkage Set, and Restricted / Back Mixing: DSMGA-II

        Shih-Huan Hsu, Tian-Li Yu

        arXiv:1807.11669

        Parameters
        ----------
        gene_index : int
            Gene which starts the sequence of dependencies.
        block_width : int
            Block width for given index. Used to speed up computation.
            By default, calculate ILS for whole genome.

        Returns
        -------
        np.ndarray
            Incremental Linkage Set
        """

        ils = [gene_index]

        if block_width is None:
            block_width = self.GENOME_SIZE

        dependencies = self.interactions[gene_index, :].copy()
        unavailable_genes = np.zeros(self.GENOME_SIZE, dtype=bool)
        unavailable_genes[gene_index] = True
        dependencies[unavailable_genes] = -1

        while len(ils) < block_width:

            max_index = np.argmax(dependencies)
            ils.append(max_index)

            unavailable_genes[max_index] = True
            dependencies += self.interactions[max_index, :]
            dependencies[unavailable_genes] = -1

        return np.array(ils[1:])

    def _get_tree(self, target_index: int) -> Tree:
        interactions = self.interactions.copy()
        eye = np.eye(self.GENOME_SIZE, dtype=bool)
        interactions[eye] = 0

        tree = Tree()
        tree.add_node(target_index)
        last_level = [target_index]

        while last_level:
            next_level = []

            for gene_index in last_level:
                gene_interactions = interactions[gene_index, :]
                positive_interactions = np.argwhere(gene_interactions > 0)
                positive_interactions = positive_interactions.squeeze(axis=1)

                leaves = set(tree.get_leaves())
                new_interactions = [
                    interaction for interaction in positive_interactions
                    if interaction not in leaves
                ]

                weights = gene_interactions[new_interactions]

                tree.add_edges_from(
                    zip(
                        [gene_index] * len(new_interactions),
                        new_interactions
                    ),
                    weight=weights
                )

                interactions[gene_index, positive_interactions] = 0
                interactions[positive_interactions, gene_index] = 0
                next_level += list(positive_interactions)

            last_level = next_level

        return tree
