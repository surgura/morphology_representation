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

    mappeds = [
        GraphAdjform(
            *model.decode(repr, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )
        for repr in reprset.representations
    ]
    mapped_as_pqgrams = [tree_to_pqgrams(mapped) for mapped in mappeds]

    solset: List[DirectedTreeNodeform]
    with open(config.GENEVALSOL_OUT(run), "rb") as f:
        solset = pickle.load(f)
    sol_as_pqgrams = [tree_to_pqgrams(sol.to_graph_adjform()) for sol in solset]

    slices = [
        (
            job_i * len(sol_as_pqgrams) // parallelism,
            (job_i + 1) * len(sol_as_pqgrams) // parallelism,
        )
        for job_i in range(parallelism)
    ]
    slices[-1] = (slices[-1][0], len(sol_as_pqgrams))
    results: List[List[float]] = joblib.Parallel(n_jobs=parallelism)(
        [
            joblib.delayed(smallest_distance_multiple)(
                sol_as_pqgrams, mapped_as_pqgrams, slice
            )
            for slice in slices
        ]
    )
    distances = sum(results, [])

    coverage = float(1.0 - np.average(distances))

    out_file = config.CVGRTGAE_OUT(run=run, t_dim=t_dim, r_dim=r_dim)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(coverage, f)


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