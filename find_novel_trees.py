"""
Find a set of trees that reasonably represent the complete space of robots using novelty and mutation.
"""

from typing import List
from tree import DirectedTreeNodeform, Node
import numpy as np
import config
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from render2d import render_modular_robot2d
from robot_rgt import tree_to_body


def make_random_tree(
    rng: np.random.Generator, grammar: tree_grammar.TreeGrammar
) -> DirectedTreeNodeform:
    tree = DirectedTreeNodeform()
    for _ in range(config.FNT_INITIAL_MUTATIONS):
        tree.mutate_binomial(
            rng,
            grammar,
            max_modules=config.FNT_MAX_MODULES,
            n=config.FNT_MUTATE_N,
            p=config.FNT_MUTATE_P,
        )
    return tree


def make_initial_population(
    rng: np.random.Generator, grammar: tree_grammar.TreeGrammar
) -> List[DirectedTreeNodeform]:
    return [make_random_tree(rng, grammar) for _ in range(config.FNT_NUM_TREES)]


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(config.FNT_RNG_SEED))
    grammar = make_body_rgt()

    population = make_initial_population(rng, grammar)
    as_adj = [r.to_graph_adjform() for r in population]
    for i, r in enumerate(as_adj):
        render_modular_robot2d(tree_to_body(r), f"novel/{i}.png")


if __name__ == "__main__":
    main()
