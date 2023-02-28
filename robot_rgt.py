from rtgae import tree_grammar
from revolve2.core.modular_robot import Body, Brick, ActiveHinge, Module, Core
from typing import List, NamedTuple
import numpy as np


def make_body_rgt() -> tree_grammar.TreeGrammar:
    alphabet = {"core": 4, "brick": 3, "active_hinge": 1, "empty": 0}
    nonterminals = ["start", "child"]
    start = "start"
    rules = {
        "start": [("core", ["child", "child", "child", "child"])],
        "child": [
            ("brick", ["child", "child", "child"]),
            ("active_hinge", ["child"]),
            ("empty", []),
        ],
    }
    grammar = tree_grammar.TreeGrammar(alphabet, nonterminals, start, rules)

    return grammar


class Tree(NamedTuple):
    nodes: List[str]
    adj: List[List[int]]


def __module_to_tree(module: Module, tree: Tree) -> None:
    if isinstance(module, Core):
        tree.nodes.append("core")
    elif isinstance(module, Brick):
        tree.nodes.append("brick")
    elif isinstance(module, ActiveHinge):
        tree.nodes.append("active_hinge")
    else:
        raise NotImplementedError()

    adj = []
    tree.adj.append(adj)

    for child in module.children:
        next_index = len(tree.nodes)
        adj.append(next_index)
        if child is None:
            tree.nodes.append("empty")
            tree.adj.append([])
        else:
            __module_to_tree(child, tree)


def body_to_tree(body: Body) -> Tree:
    """
    Convert a body to a tree.

    :param body: The body to convert.
    :returns: The created tree.
    """

    tree = Tree(nodes=[], adj=[])
    __module_to_tree(body.core, tree)

    return tree


def __children_to_modules(node_index: int, tree: Tree, parent_module: Module) -> None:
    for i, child_node_index in enumerate(tree.adj[node_index]):
        if tree.nodes[child_node_index] == "brick":
            parent_module.children[i] = Brick(0.0)
            __children_to_modules(child_node_index, tree, parent_module.children[i])
        elif tree.nodes[child_node_index] == "active_hinge":
            parent_module.children[i] = ActiveHinge(0.0)
            __children_to_modules(child_node_index, tree, parent_module.children[i])
        elif (
            tree.nodes[child_node_index] == "empty"
            or tree.nodes[child_node_index] == "child"
        ):
            parent_module.children[i] = None
        else:
            raise NotImplementedError()


def tree_to_body(tree: Tree) -> Body:
    """
    Convert a tree to a body.

    :param tree: The tree to convert.
    :returns: The created body.
    """

    body = Body()
    __children_to_modules(0, tree, body.core)
    body.finalize()
    return body
