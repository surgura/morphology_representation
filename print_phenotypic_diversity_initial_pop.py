import logging
import argparse
import indices_range
import config
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from robot_rgt import make_body_rgt, body_to_tree
import torch
from evaluation_representation_set import EvaluationRepresentationSet
import pickle
from tree import GraphAdjform
import pathlib
import math
import hashlib
from torch.nn.functional import normalize
from apted_util import tree_to_apted, apted_tree_edit_distance
import robot_optimization.cmaes.model as cmodel
from revolve2.core.database import open_database_sqlite
import robot_optimization.benchmark.model as bmodel
import pandas
from sqlalchemy.orm import Session
from sqlalchemy import select
from opt_robot_displacement_cmaes import representation_to_body
from typing import List
import joblib
import numpy as np
import numpy.typing as npt
import apted.helpers
import matplotlib.pyplot as plt


def plot_with_error_bars(df, y_col, color, ax, label):
    aggregated = df.groupby("performed_evaluations").agg(["mean", "std"])
    means = aggregated[y_col]["mean"]
    stds = aggregated[y_col]["std"]

    means.plot(y=y_col, ax=ax, color=color, label=label)
    ax.fill_between(means.index, means - stds, means + stds, color=color, alpha=0.2)


def cppn(experiment_name: str, run: int, ax=None, color="teal") -> None:
    all = []

    for optrun in range(config.ROBOPT_RUNS):
        in_file = config.PHENDIV_CPPN_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
        )
        with open(in_file, "rb") as f:
            distance_matrices: List[npt.NDArray[np.float64]] = pickle.load(f)

        print(distance_matrices[0])
        diversities = [float(np.mean(matrix)) for matrix in distance_matrices]

        print(f"CPPN {run=} {optrun=}")
        print(f"{diversities[0]}")

        all.append(diversities[0])

    print(f"mean: {np.mean(all)}")
    print(f"std: {np.std(all)}")


def cmaes(
    experiment_name: str,
    run: int,
    t_dim: int,
    r_dim: int,
    margin: float,
    gain: float,
    ax=None,
    color="teal",
) -> None:
    all = []

    for optrun in range(config.ROBOPT_RUNS):
        in_file = config.PHENDIV_CMAES_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
            t_dim=t_dim,
            r_dim=r_dim,
            margin=margin,
            gain=gain,
        )
        with open(in_file, "rb") as f:
            distance_matrices: List[npt.NDArray[np.float64]] = pickle.load(f)

        diversities = [float(np.mean(matrix)) for matrix in distance_matrices]

        print(f"CMAES {run=} {optrun=} {t_dim=} {r_dim=} {margin=} {gain=}")
        print(f"{diversities[0]}")

        all.append(diversities[0])

    print(f"mean: {np.mean(all)}")
    print(f"std: {np.std(all)}")


def vector_sample(
    experiment_name: str,
    run: int,
    t_dim: int,
    r_dim: int,
    margin: float,
    gain: float,
    ax=None,
    color="teal",
) -> None:
    in_file = config.PHENDIV_VECTOR_OUT(
        experiment_name=experiment_name,
        run=run,
        t_dim=t_dim,
        r_dim=r_dim,
        margin=margin,
        gain=gain,
    )
    with open(in_file, "rb") as f:
        distance_matrices: List[npt.NDArray[np.float64]] = pickle.load(f)

    print(distance_matrices[0])
    diversities = [float(np.mean(matrix)) for matrix in distance_matrices]

    print(f"Random vector sample {run=} {t_dim=} {r_dim=} {margin=} {gain=}")
    print(f"{diversities[0]}")


def together(
    experiment_name: str,
    run: int,
) -> None:
    excmaes = "paper_nometric"
    excppn = "papercppn"

    for t_dim in config.MODEL_T_DIMS:
        for r_dim in config.MODEL_R_DIMS:
            for margin in config.TRAIN_DD_MARGINS:
                for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                    vector_sample(
                        experiment_name=excmaes,
                        run=run,
                        t_dim=t_dim,
                        r_dim=r_dim,
                        margin=margin,
                        gain=gain,
                    )
                    print()
                    cmaes(
                        experiment_name=excmaes,
                        run=run,
                        t_dim=t_dim,
                        r_dim=r_dim,
                        margin=margin,
                        gain=gain,
                    )
    print()
    cppn(experiment_name=excppn, run=run)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)

    args = parser.parse_args()

    for run in range(config.RUNS):
        together(experiment_name=args.experiment_name, run=run)


if __name__ == "__main__":
    main()
