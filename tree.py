from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

import numpy as np

from rtgae import tree_grammar


@dataclass
class GraphAdjform:
    nodes: List[str]
    adj: List[List[int]]


@dataclass
class Node:
    data: str
    parent: Optional[Node]
    parent_index: Optional[int]
    children: List[Optional[Node]]


class DirectedTreeNodeform:
    root: Node
    __num_nodes: int
    __nodes_with_none_children: List[Node]

    def __init__(self) -> None:
        """
        Initialize this object.

        :param root: The root node of the tree.
        """
        self.root = Node("core", None, None, [None] * 4)
        self.__num_nodes = 1
        self.__nodes_with_none_children = [self.root]

    def copy(self) -> DirectedTreeNodeform:
        cp = copy.deepcopy(self)
        return copy.deepcopy(self)

    @classmethod
    def __module_to_tree(cls, node: Node, tree: GraphAdjform) -> None:
        tree.nodes.append(node.data)

        adj: List[int] = []
        tree.adj.append(adj)

        for child in node.children:
            next_index = len(tree.nodes)
            adj.append(next_index)
            if child is None:
                tree.nodes.append("empty")
                tree.adj.append([])
            else:
                cls.__module_to_tree(child, tree)

    def to_graph_adjform(self) -> GraphAdjform:
        tree = GraphAdjform(nodes=[], adj=[])
        self.__module_to_tree(self.root, tree)
        return tree

    def mutate_binomial(
        self,
        rng: np.random.Generator,
        grammar: tree_grammar.TreeGrammar,
        max_modules: int,
        n: int = 5,
        p: float = 0.5,
    ) -> None:
        """
        Mutate the provided tree in-place a number of times as sampled from a binomial distribution.

        :param rng: Random number generator.
        :param grammar: Grammar that the tree adheres to.
        :param n: Maximum number of mutations.
        :param p: Chance to add another mutation.
        """
        for _ in range(rng.binomial(n=n, p=p)):
            self.mutate(rng, grammar, max_modules)

    def mutate(
        self,
        rng: np.random.Generator,
        grammar: tree_grammar.TreeGrammar,
        max_modules: int,
    ) -> None:
        """
        Mutate the provided tree in-place by adding or removing a leaf.

        :param rng: Random number generator.
        :param grammar: Grammar that the tree adheres to.
        """
        can_remove = [
            node
            for node in self.__nodes_with_none_children
            if all([child is None for child in node.children])
            and node.parent is not None
        ]
        can_fill: List[Tuple[Node, int]] = []
        for node in self.__nodes_with_none_children:
            for i, child in enumerate(node.children):
                if child is None:
                    can_fill.append((node, i))

        while True:
            fillorremove = rng.random() < 0.5
            if (fillorremove and len(can_fill) != 0) or (
                not fillorremove and len(can_remove) != 0
            ):
                break
        if fillorremove:  # fill a node
            choice = rng.integers(len(can_fill))
            if self.__num_nodes < max_modules:
                parent, index = can_fill[choice]
                choice2 = rng.integers(0, 2)
                if choice2 == 0:  # brick
                    child = Node(
                        "brick", parent=parent, parent_index=index, children=[None] * 3
                    )
                else:
                    child = Node(
                        "active_hinge",
                        parent=parent,
                        parent_index=index,
                        children=[None],
                    )
                parent.children[index] = child
                self.__nodes_with_none_children.append(child)
                if all([child is not None for child in parent.children]):
                    self.__nodes_with_none_children.remove(parent)
                self.__num_nodes += 1
        else:  # remove a node:
            choice = rng.integers(len(can_remove))
            node = can_remove[choice]
            assert node.parent is not None, "cannot remove root"
            assert node.parent_index is not None
            self.__nodes_with_none_children.remove(node)
            node.parent.children[node.parent_index] = None
            if node.parent not in self.__nodes_with_none_children:
                self.__nodes_with_none_children.append(node.parent)
            self.__num_nodes -= 1
