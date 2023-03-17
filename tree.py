from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Set, Any
from rtgae import tree_grammar
from dataclasses import dataclass
import copy


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

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Node), "can only compare Node with Node"

        return self.data == other.data and all(
            [
                (child1 is None and child2 is None)
                or (child1 is not None and child2 is not None and child1 == child2)
                for child1, child2 in zip(self.children, other.children)
            ]
        )


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
        Mutate the provided tree in-place by adding or removing a leave.

        :param rng: Random number generator.
        :param grammar: Grammar that the tree adheres to.
        """
        can_remove = [
            node
            for node in self.__nodes_with_none_children
            if all([child is None for child in node.children])
        ]
        can_fill: List[Tuple[Node, int]] = []
        for node in self.__nodes_with_none_children:
            for i, child in enumerate(node.children):
                if child is None:
                    can_fill.append((node, i))

        choice = rng.integers(0, len(can_remove) + len(can_fill))
        if choice >= len(can_remove):  # fill a node
            if self.__num_nodes < max_modules:
                parent, index = can_fill[choice - len(can_remove)]
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
            node = can_remove[choice]
            if node.parent is not None:  # cannot remove root
                assert node.parent_index is not None
                self.__nodes_with_none_children.remove(node)
                node.parent.children[node.parent_index] = None
                if node.parent not in self.__nodes_with_none_children:
                    self.__nodes_with_none_children.append(node.parent)
                self.__num_nodes -= 1

    def __eq__(self, other: Any) -> bool:
        assert isinstance(
            other, DirectedTreeNodeform
        ), "can only compare DirectedTreeNodeform with DirectedTreeNodeform"

        return self.root == other.root
