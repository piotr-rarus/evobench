import numpy as np
from plotly.graph_objects import Figure
from pytest import fixture

from evobench.discrete import Trap

from ..tree import Tree


@fixture()
def trap() -> Trap:
    return Trap(blocks=[3] * 2)


def test_ils(trap: Trap):
    ils = trap.dsm.ils

    assert isinstance(ils, list)
    assert all(isinstance(gene_ils, np.ndarray) for gene_ils in ils)
    assert all(gene_ils.size < trap.genome_size for gene_ils in ils)


def test_trees(trap: Trap):
    trees = trap.dsm.trees

    assert isinstance(trees, list)
    assert all(isinstance(tree, Tree) for tree in trees)
    assert all(tree.get_depth() == 2 for tree in trees)


def test_levels(trap: Trap):
    levels = trap.dsm.levels
    assert isinstance(levels, np.ndarray)
    assert levels.shape == trap.dsm.interactions.shape


def test_fig(trap: Trap):
    fig = trap.dsm.get_fig()
    assert isinstance(fig, Figure)


def test_levels_fig(trap: Trap):
    figs = [tree.get_levels_fig() for tree in trap.dsm.trees]

    assert isinstance(figs, list)
    assert all(isinstance(fig, Figure) for fig in figs)


def test_tree_fig(trap: Trap):
    figs = [tree.get_fig() for tree in trap.dsm.trees]

    assert isinstance(figs, list)
    assert all(isinstance(fig, Figure) for fig in figs)
