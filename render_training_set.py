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
from tree import DirectedTreeNodeform
from train_set import TrainSet


def do_run(run: int, experiment_name: str) -> None:
    trainset = TrainSet(run=run, experiment_name=experiment_name)

    for i, tree in enumerate(trainset._graph_adj_form):
        out_file = config.RENDERTRAIN_OUT(
            run=run, experiment_name=experiment_name, item_i=i
        )
        pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        render_modular_robot_radial(tree_to_body(tree), out_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument(
        "-r",
        "--runs",
        type=indices_range.indices_type(range(config.RUNS)),
        required=True,
    )
    args = parser.parse_args()

    for run in args.runs:
        do_run(run=run, experiment_name=args.experiment_name)


if __name__ == "__main__":
    main()
