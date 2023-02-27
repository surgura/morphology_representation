from rtgae import tree_grammar
from revolve2.core.modular_robot import Body, Brick, ActiveHinge, Module, Core
from typing import List, NamedTuple


def make_body_rgt() -> tree_grammar.TreeGrammar:
    alphabet = {
        "empty": 0,
        "core_children": 4,
        "brick_children": 3,
        "active_hinge_children": 1,
        "brick": 1,
        "active_hinge": 1,
        "core": 0,
    }
    nonterminals = ["core", "brick", "active_hinge", "child"]
    start = "core"
    rules = {
        "core": [("core_children", ["child", "child", "child", "child"])],
        "brick": [("brick_children", ["child", "child", "child"])],
        "active_hinge": [("active_hinge_children", ["child"])],
        "child": [
            ("empty", []),
            ("brick", ["brick"]),
            ("active_hinge", ["active_hinge"]),
        ],
    }
    grammar = tree_grammar.TreeGrammar(alphabet, nonterminals, start, rules)

    return grammar


class Tree(NamedTuple):
    nodes: List[str]
    adj: List[List[int]]


# def __module_to_tree(module: Module, tree: Tree) -> None:
#     if isinstance(module, Core):
#         tree.nodes.append("core")
#         tree.nodes.append("core_children")
#     elif isinstance(module, Brick):
#         tree.nodes.append("brick")
#         tree.nodes.append("brick_children")
#     elif isinstance(module, ActiveHinge):
#         tree.nodes.append("active_hinge")
#         tree.nodes.append("active_hinge_children")
#     else:
#         raise NotImplementedError()

#     tree.adj.append([len(tree.nodes) - 1])
#     adj = []
#     tree.adj.append(adj)

#     for child in module.children:
#         next_index = len(tree.nodes)
#         adj.append(next_index)
#         if child is None:
#             tree.nodes.append("empty")
#             tree.adj.append([])
#         else:
#             __module_to_tree(child, tree)


def __module_to_tree(module: Module, tree: Tree) -> None:
    if isinstance(module, Core):
        tree.nodes.append("core_children")
    elif isinstance(module, Brick):
        tree.nodes.append("brick_children")
    elif isinstance(module, ActiveHinge):
        tree.nodes.append("active_hinge_children")
    else:
        raise NotImplementedError()

    tree.adj.append([len(tree.nodes) - 1])
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
    """ "
    Convert a body to a tree.

    :param body: The body to convert.
    :returns: The created tree.
    """

    tree = Tree(nodes=[], adj=[])
    __module_to_tree(body.core, tree)

    return tree
