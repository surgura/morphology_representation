"""
Find a set of trees that reasonably represent the complete space of robots using novelty and mutation.
"""

import argparse
import hashlib
import logging
import pathlib
import pickle
from typing import List, Tuple

import numpy as np

import config
import indices_range
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from tree import DirectedTreeNodeform
import torch
from apted_util import tree_to_apted, apted_tree_edit_distance
import joblib
import apted.helpers


def compute_distance_matrix_part(
    asapted: List[apted.helpers.Tree], slice: Tuple[int, int]
) -> torch.Tensor:
    distance_matrix_part = torch.zeros((slice[1] - slice[0], len(asapted)))

    for i, slice_i in enumerate(range(slice[0], slice[1])):
        print(i)
        for j, other in enumerate(asapted):
            distance_matrix_part[i, j] = apted_tree_edit_distance(
                asapted[slice_i], asapted[j]
            )
    return distance_matrix_part


def compute_distance_matrix(
    trees: List[DirectedTreeNodeform], parallelism: int
) -> torch.Tensor:
    asapted = [tree_to_apted(tree.to_graph_adjform()) for tree in trees]

    slices = [
        (
            job_i * len(asapted) // parallelism,
            (job_i + 1) * len(asapted) // parallelism,
        )
        for job_i in range(parallelism)
    ]
    slices[-1] = (slices[-1][0], len(asapted))

    results: List[torch.Tensor] = joblib.Parallel(n_jobs=parallelism)(
        [
            joblib.delayed(compute_distance_matrix_part)(asapted, slice)
            for slice in slices
        ]
    )

    return torch.cat(results)


def do_run(
    run: int, experiment_name: str, parallelism: int, grammar: tree_grammar.TreeGrammar
) -> None:
    rng_seed = int(
        hashlib.sha256(
            f"generate_training_set_seed{config.GENTRAIN_RNG_SEED}_run{run}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    archive: List[DirectedTreeNodeform] = sum(
        [
            [
                DirectedTreeNodeform.random_uniform(size, rng)
                for _ in range(
                    config.GENTRAIN_ARCHIVE_SIZE // (config.MODEL_MAX_MODULES + 1 - 1)
                )
            ]
            for size in range(1, config.MODEL_MAX_MODULES + 1)
        ],
        [],
    )
    distance_matrix = compute_distance_matrix(archive, parallelism)
    out_file = config.GENTRAIN_OUT(run=run, experiment_name=experiment_name)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump({"trees": archive, "distance_matrix": distance_matrix}, f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallelism", type=int, required=True)
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument(
        "-r",
        "--runs",
        type=indices_range.indices_type(range(config.RUNS)),
        required=True,
    )
    args = parser.parse_args()

    grammar = make_body_rgt()

    for run in args.runs:
        do_run(
            run=run,
            experiment_name=args.experiment_name,
            parallelism=args.parallelism,
            grammar=grammar,
        )


if __name__ == "__main__":
    main()
