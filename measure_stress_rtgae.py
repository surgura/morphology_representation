import logging
import argparse
import indices_range
import config
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from robot_rgt import make_body_rgt
import torch
from evaluation_representation_set import EvaluationRepresentationSet
import pickle
from tree import GraphAdjform
from pqgrams_util import tree_to_pqgrams
from tree import DirectedTreeNodeform
from typing import List, Tuple
import pqgrams
import numpy as np
import joblib
import pathlib
import math


def smallest_distance(
    tree: pqgrams.Profile, compare_to: List[pqgrams.Profile]
) -> float:
    return [tree.edit_distance(other) for other in compare_to]


def smallest_distance_multiple(
    trees: List[pqgrams.Profile],
    compare_to: List[pqgrams.Profile],
    slice: Tuple[int, int],
) -> List[float]:
    return [smallest_distance(tree, compare_to) for tree in trees[slice[0] : slice[1]]]


def do_run(run: int, t_dim_i: int, r_dim_i: int, parallelism: int) -> None:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    logging.info(f"Measuring coverage for RTGAE {run=} {t_dim=} {r_dim=}")

    grammar = make_body_rgt()
    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(config.TRAIN_OUT(run=run, t_dim=t_dim, r_dim=r_dim))
    )

    reprset: EvaluationRepresentationSet[torch.Tensor]
    with open(config.GENEVALREPR_OUT_RTGAE(run, t_dim, r_dim), "rb") as f:
        reprset = pickle.load(f)

    # normalized stress or kruskal's stress-1
    # S = sqrt[ ( sum over i,j (d_ij(X) - d_ij(Y))^2 ) / ( sum over i,j (d_ij(X))^2 ) ]

    repr_mapped_as_pqgrams = {
        repr: tree_to_pqgrams(
            GraphAdjform(
                *model.decode(repr, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
            )
        )
        for repr in reprset.representations
    }

    dists_in_solspace = [
        repr_mapped_as_pqgrams[a].edit_distance(repr_mapped_as_pqgrams[b])
        for (a, b) in reprset.pairs
    ]

    stress = math.sqrt(
        sum(
            [
                (dist_repr - dist_sol) ** 2
                for dist_repr, dist_sol in zip(reprset.distances, dists_in_solspace)
            ]
        )
        / sum([dist_repr**2 for dist_repr in reprset.distances])
    )

    dist_pairs = list(zip(reprset.distances, dists_in_solspace))

    out_file = config.STRESSRTGAE_OUT(run=run, t_dim=t_dim, r_dim=r_dim)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump({"stress": stress, "dist_pairs": dist_pairs}, f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallelism", type=int, default=1)
    parser.add_argument(
        "-r",
        "--runs",
        type=indices_range.indices_type(range(config.RUNS)),
        required=True,
    )
    parser.add_argument(
        "--r_dims",
        type=indices_range.indices_type(range(len(config.MODEL_R_DIMS))),
        required=True,
    )
    parser.add_argument(
        "--t_dims",
        type=indices_range.indices_type(range(len(config.MODEL_T_DIMS))),
        required=True,
    )
    args = parser.parse_args()

    for run in args.runs:
        for t_dim_i in args.t_dims:
            for r_dim_i in args.r_dims:
                do_run(
                    run=run,
                    t_dim_i=t_dim_i,
                    r_dim_i=r_dim_i,
                    parallelism=args.parallelism,
                )


if __name__ == "__main__":
    main()
