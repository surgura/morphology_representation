"""
Find a set of trees that reasonably represent the complete space of robots using novelty and mutation.
"""

from typing import List, Tuple
from tree import DirectedTreeNodeform
import numpy as np
import config
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from render2d import render_modular_robot2d
from robot_rgt import tree_to_body
from pqgrams_util import tree_to_pqgrams
import pqgrams
import joblib


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
    return [make_random_tree(rng, grammar) for _ in range(config.FNT_POPULATION_SIZE)]


def compare_trees(left: pqgrams.Profile, right: pqgrams.Profile) -> float:
    if left is right:
        return 0.0

    return left.edit_distance(right)


def measure_population_parallel(
    population: List[pqgrams.Profile], slice: Tuple[int, int]
) -> List[float]:
    return [
        sum([compare_trees(population[i], other_tree) for other_tree in population])
        for i in range(slice[0], slice[1])
    ]


def measure_population(
    population: List[DirectedTreeNodeform], num_jobs: int
) -> List[float]:
    as_pqgrams = [tree_to_pqgrams(tree.to_graph_adjform()) for tree in population]
    slices = [
        (
            job_i * len(as_pqgrams) // num_jobs,
            (job_i + 1) * len(as_pqgrams) // num_jobs,
        )
        for job_i in range(num_jobs)
    ]
    slices[-1] = (slices[-1][0], len(as_pqgrams))
    results = joblib.Parallel(n_jobs=num_jobs)(
        [
            joblib.delayed(measure_population_parallel)(
                as_pqgrams,
                slice,
            )
            for slice in slices
        ]
    )
    return sum(results, [])


def next_generation(
    rng: np.random.Generator,
    grammar: tree_grammar.TreeGrammar,
    population: List[DirectedTreeNodeform],
    num_jobs: int,
) -> Tuple[List[DirectedTreeNodeform], List[float]]:
    children = [
        population[rng.integers(0, len(population))].copy()
        for _ in range(config.FNT_OFFSPRING_SIZE)
    ]
    for child in children:
        child.mutate_binomial(
            rng,
            grammar,
            max_modules=config.FNT_MAX_MODULES,
            n=config.FNT_MUTATE_N,
            p=config.FNT_MUTATE_P,
        )

    combined = population + children
    measures = measure_population(combined, num_jobs)
    best_indices = np.argsort(measures)[config.FNT_POPULATION_SIZE :]

    return [combined[i] for i in best_indices], [measures[i] for i in best_indices]


def main() -> None:
    NUM_JOBS = 8

    rng = np.random.Generator(np.random.PCG64(config.FNT_RNG_SEED))
    grammar = make_body_rgt()

    population = make_initial_population(rng, grammar)

    for gen_i in range(100):
        population, fitnesses = next_generation(rng, grammar, population, NUM_JOBS)
        print(f"Generation {gen_i}. Fitness: {sum(fitnesses)}")

    # as_adj = [r.to_graph_adjform() for r in population]
    # for i, r in enumerate(as_adj):
    #     render_modular_robot2d(tree_to_body(r), f"novel/{i}.png")


if __name__ == "__main__":
    main()
