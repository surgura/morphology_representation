import logging
import argparse
import indices_range
import config
import joblib
import hashlib
import numpy as np
import torch
from robot_rgt import make_body_rgt
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from tree import GraphAdjform
from pqgrams_util import tree_to_pqgrams
import pickle
import pathlib
import math
from typing import List, Tuple


def measure_pair(
    model: TreeGrammarAutoEncoder, repr_1: torch.Tensor, repr_2: torch.Tensor
) -> float:
    repr_dist = torch.norm(repr_1 - repr_2).item() / math.sqrt(repr_1.size(0))

    sol_1 = GraphAdjform(*model.decode(repr_1, max_size=32)[:2])
    sol_2 = GraphAdjform(*model.decode(repr_2, max_size=32)[:2])
    sol_dist = tree_to_pqgrams(sol_1).edit_distance(tree_to_pqgrams(sol_2))

    return abs(repr_dist - sol_dist)


def do_run(run: int, t_dim_i: int, r_dim_i: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    logging.info(f"Measuring run{run} t_dim{t_dim} r_dim{r_dim}")

    grammar = make_body_rgt()

    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(config.TRAIN_OUT(run=run, t_dim=t_dim, r_dim=r_dim))
    )

    evset: List[Tuple[torch.Tensor, torch.Tensor]]
    with open(config.DDEVSET_OUT(run, r_dim), "rb") as f:
        evset = pickle.load(f)

    distance_distortion = np.average(
        [measure_pair(model, pair[0], pair[1]) for pair in evset]
    )

    out_dir = config.MDD_OUT(run, t_dim, r_dim)
    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir, "wb") as f:
        pickle.dump(distance_distortion, f)


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

    jobs = []
    for run in args.runs:
        for t_dim_i in args.t_dims:
            for r_dim_i in args.r_dims:
                jobs.append(
                    joblib.delayed(do_run)(run=run, t_dim_i=t_dim_i, r_dim_i=r_dim_i)
                )

    joblib.Parallel(n_jobs=args.parallelism)(jobs)


if __name__ == "__main__":
    main()
