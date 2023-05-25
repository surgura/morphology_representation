"""
Render images of the trees in the training set.
"""

import argparse
import logging
import pathlib
import pickle
from typing import List

import config
import indices_range
from render2d import render_modular_robot_radial
from robot_rgt import tree_to_body
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from robot_rgt import make_body_rgt
import torch
from tree import GraphAdjform


def do_run(
    experiment_name: str, run: int, t_dim_i: int, r_dim_i: int, parallelism: int
) -> None:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    grammar = make_body_rgt()
    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(
            config.TRAIN_OUT(
                experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
            )
        )
    )

    for i in range(100):
        repr = torch.normal(torch.zeros(r_dim), 0.8 * torch.ones(r_dim))
        # repr = torch.zeros(r_dim)
        # repr[2] += i * 0.01
        center = GraphAdjform(
            *model.decode(repr, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )
        out_file = config.SAMPLEREPR_OUT_CENTER(
            experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim, i=i
        )
        pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        render_modular_robot_radial(tree_to_body(center), out_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
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
                    experiment_name=args.experiment_name,
                    run=run,
                    t_dim_i=t_dim_i,
                    r_dim_i=r_dim_i,
                    parallelism=args.parallelism,
                )


if __name__ == "__main__":
    main()
