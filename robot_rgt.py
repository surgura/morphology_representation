from typing import List

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module
from rtgae import tree_grammar
from tree import GraphAdjform
import numpy as np
import math


def make_body_rgt() -> tree_grammar.TreeGrammar:
    """
    Make the regular tree grammar for modular robots.

    :returns: The regular tree grammar.
    """

    alphabet = ["core", "brick", "active_hinge_v", "active_hinge_h", "empty"]
    nonterminals = ["start", "child"]
    start = "start"
    rules = {
        "start": [("core", ["child", "child", "child", "child"])],
        "child": [
            ("brick", ["child", "child", "child"]),
            ("active_hinge_v", ["child"]),
            ("active_hinge_h", ["child"]),
            ("empty", []),
        ],
    }
    grammar = tree_grammar.TreeGrammar(
        alphabet, nonterminals, start, rules
    )  # type: ignore

    return grammar


def __module_to_tree(module: Module, tree: GraphAdjform) -> None:
    if isinstance(module, Core):
        tree.nodes.append("core")
    elif isinstance(module, Brick):
        tree.nodes.append("brick")
    elif isinstance(module, ActiveHinge):
        if np.isclose(module.rotation, 0.0):
            tree.nodes.append("active_hinge_v")
        elif np.isclose(module.rotation, math.pi / 2.0):
            tree.nodes.append("active_hinge_h")
        else:
            raise NotImplementedError("Rotation not as expected")
    else:
        raise NotImplementedError()

    adj: List[int] = []
    tree.adj.append(adj)

    for child in module.children:
        next_index = len(tree.nodes)
        adj.append(next_index)
        if child is None:
            tree.nodes.append("empty")
            tree.adj.append([])
        else:
            __module_to_tree(child, tree)


def body_to_tree(body: Body) -> GraphAdjform:
    """
    Convert a body to a tree.

    :param body: The body to convert.
    :returns: The created tree.
    """

    tree = GraphAdjform(nodes=[], adj=[])
    __module_to_tree(body.core, tree)

    return tree


def __children_to_modules(
    node_index: int, tree: GraphAdjform, parent_module: Module
) -> None:
    for i, child_node_index in enumerate(tree.adj[node_index]):
        if tree.nodes[child_node_index] == "brick":
            child1 = Brick(0.0)
            parent_module.children[i] = child1
            __children_to_modules(child_node_index, tree, child1)
        elif tree.nodes[child_node_index] == "active_hinge_v":
            child2 = ActiveHinge(0.0)
            parent_module.children[i] = child2
            __children_to_modules(child_node_index, tree, child2)
        elif tree.nodes[child_node_index] == "active_hinge_h":
            child2 = ActiveHinge(math.pi / 2.0)
            parent_module.children[i] = child2
            __children_to_modules(child_node_index, tree, child2)
        elif (
            tree.nodes[child_node_index] == "empty"
            or tree.nodes[child_node_index] == "child"
        ):
            parent_module.children[i] = None
        else:
            raise NotImplementedError()


def tree_to_body(tree: GraphAdjform) -> Body:
    """
    Convert a tree to a body.

    :param tree: The tree to convert.
    :returns: The created body.
    """

    body = Body()
    __children_to_modules(0, tree, body.core)
    body.finalize()
    return body
