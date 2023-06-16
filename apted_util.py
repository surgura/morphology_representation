import apted.helpers

from tree import GraphAdjform
from apted import APTED


def __tree_to_apted_children(tree: GraphAdjform, node_i: int) -> str:
    return str.join(
        "",
        [f"{{{__tree_to_apted_recur(tree, child_i)}}}" for child_i in tree.adj[node_i]],
    )


def __tree_to_apted_recur(tree: GraphAdjform, node_i: int) -> str:
    if len(tree.adj[node_i]) > 0:
        return f"{tree.nodes[node_i]}{__tree_to_apted_children(tree, node_i)}"
    else:
        return tree.nodes[node_i]


def tree_to_apted(tree: GraphAdjform) -> apted.helpers.Tree:
    return apted.helpers.Tree.from_text(f"{{{__tree_to_apted_recur(tree, 0)}}}")


def apted_tree_edit_distance(
    tree1: apted.helpers.Tree, tree2: apted.helpers.Tree
) -> int:
    return APTED(tree1, tree2).compute_edit_distance()
