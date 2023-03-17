import config
import pickle
from tree import DirectedTreeNodeform
from typing import List, Tuple
import numpy as np
from pqgrams_util import tree_to_pqgrams
import pqgrams
import joblib
import render2d
import robot_rgt


def rank_trees_by_distance_parallel(
    tree: pqgrams.Profile,
    population: List[DirectedTreeNodeform],
    slice: Tuple[int, int],
) -> List[DirectedTreeNodeform]:
    return [
        tree.edit_distance(tree_to_pqgrams(other_tree.to_graph_adjform()))
        for other_tree in population[slice[0] : slice[1]]
    ]


def rank_trees_by_distance(
    tree: DirectedTreeNodeform, population: List[DirectedTreeNodeform], num_jobs: int
) -> List[Tuple[DirectedTreeNodeform, float]]:
    """
    Rank the trees based on their distance to the given tree.

    Uses the pqgrams algorithm.

    :param tree: The tree to compare with.
    :param population: A population of trees.
    """
    tree_as_pqgrams = tree_to_pqgrams(tree.to_graph_adjform())

    slices = [
        (
            job_i * len(population) // num_jobs,
            (job_i + 1) * len(population) // num_jobs,
        )
        for job_i in range(num_jobs)
    ]

    results = joblib.Parallel(n_jobs=num_jobs)(
        [
            joblib.delayed(rank_trees_by_distance_parallel)(
                tree_as_pqgrams,
                population,
                slice,
            )
            for slice in slices
        ]
    )
    distances = sum(results, [])

    return [(population[i], distances[i]) for i in np.argsort(distances)]


def main() -> None:
    NUM_JOBS = 1
    RNG_SEED = 0

    rng = np.random.Generator(np.random.PCG64(RNG_SEED))

    with open(config.FNT_BEST, "rb") as file:
        best_pop: List[DirectedTreeNodeform]
        (best_pop, _, _) = pickle.load(file)

    unique = []
    for item in best_pop:
        if item not in unique:
            unique.append(item)

    random_sample = unique[rng.integers(0, len(unique))]
    ranked = rank_trees_by_distance(random_sample, unique, num_jobs=NUM_JOBS)

    for i, (tree, _) in enumerate(ranked):
        render2d.render_modular_robot2d(
            robot_rgt.tree_to_body(tree.to_graph_adjform()), f"img/{i}.png"
        )


if __name__ == "__main__":
    main()
