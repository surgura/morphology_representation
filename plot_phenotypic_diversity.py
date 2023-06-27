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
    dfs = []
    for optrun in range(config.ROBOPT_RUNS):
        in_file = config.PHENDIV_CPPN_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
        )
        with open(in_file, "rb") as f:
            distance_matrices: List[npt.NDArray[np.float64]] = pickle.load(f)

        diversities = [float(np.mean(matrix)) for matrix in distance_matrices]
        df = pandas.DataFrame.from_records(
            zip(
                [
                    i * config.OPTCMAES_NUM_EVALUATIONS // config.ROBOPT_NUM_GENERATIONS
                    for i in range(len(diversities))
                ],
                diversities,
            ),
            columns=["performed_evaluations", "diversity"],
        )
        dfs.append(df)

    df = pandas.concat(dfs)

    save = ax is None
    if ax is None:
        ax = plt.subplot()
    plot_with_error_bars(
        df[["performed_evaluations", "diversity"]], "diversity", color, ax, "CPPN"
    )

    if save:
        ax.set_ylabel("Diversity")
        ax.set_xlabel("Evaluations")
        ax.set_ylim(bottom=config.PLTPHENDIV_Y[0], top=config.PLTPHENDIV_Y[1])

        out_dir = config.PLTPHENDIV_CPPN_OUT(
            experiment_name=experiment_name,
            run=run,
        )
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir, bbox_inches="tight")
        plt.close()


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
    dfs = []
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

        dbengine = open_database_sqlite(
            config.OPTCMAES_OUT(
                experiment_name=experiment_name,
                run=run,
                optrun=optrun,
                t_dim=t_dim,
                r_dim=r_dim,
                margin=margin,
                gain=gain,
            )
        )
        with Session(dbengine) as ses:
            stmt = select(
                cmodel.Generation.performed_evaluations,
            )
            performed_evaluations = [x for x in ses.execute(stmt).scalars()]

        df = pandas.DataFrame.from_records(
            zip(performed_evaluations, diversities),
            columns=["performed_evaluations", "diversity"],
        )
        dfs.append(df)

    df = pandas.concat(dfs)

    save = ax is None
    if ax is None:
        ax = plt.subplot()
    plot_with_error_bars(
        df[["performed_evaluations", "diversity"]], "diversity", color, ax, "Vector"
    )

    if save:
        ax.set_ylabel("Diversity")
        ax.set_xlabel("Evaluations")
        ax.set_ylim(bottom=config.PLTPHENDIV_Y[0], top=config.PLTPHENDIV_Y[1])

        out_dir = config.PLTPHENDIV_CMAES_OUT(
            experiment_name=experiment_name,
            run=run,
            t_dim=t_dim,
            r_dim=r_dim,
            margin=margin,
            gain=gain,
        )
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir, bbox_inches="tight")
        plt.close()


def together(
    experiment_name: str,
    run: int,
) -> None:
    excmaes = "paper_nometric"
    excppn = "papercppn"

    ax = plt.subplot()

    for t_dim in config.MODEL_T_DIMS:
        for r_dim in config.MODEL_R_DIMS:
            for margin in config.TRAIN_DD_MARGINS:
                for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                    cmaes(
                        experiment_name=excmaes,
                        run=run,
                        t_dim=t_dim,
                        r_dim=r_dim,
                        margin=margin,
                        gain=gain,
                        ax=ax,
                        color="sandybrown",
                    )
    cppn(experiment_name=excppn, run=run, ax=ax, color="teal")

    ax.set_ylabel("Phenotypic diversity")
    ax.set_xlabel("Evaluations")
    ax.set_ylim(bottom=config.PLTPHENDIV_Y[0], top=config.PLTPHENDIV_Y[1])
    plt.legend(loc="upper right")

    out_dir = config.PLTPHENDIV_TOGETHER_OUT(experiment_name=experiment_name, run=run)
    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir, bbox_inches="tight")
    plt.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument(
        "--repr", type=str, required=True, choices=["cmaes", "cppn", "together"]
    )

    args = parser.parse_args()

    if args.repr == "cmaes":
        for run in range(config.RUNS):
            for t_dim in config.MODEL_T_DIMS:
                for r_dim in config.MODEL_R_DIMS:
                    for margin in config.TRAIN_DD_MARGINS:
                        for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                            cmaes(
                                experiment_name=args.experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                                margin=margin,
                                gain=gain,
                            )
    elif args.repr == "cppn":
        for run in range(config.RUNS):
            cppn(
                experiment_name=args.experiment_name,
                run=run,
            )
    elif args.repr == "together":
        for run in range(config.RUNS):
            together(experiment_name=args.experiment_name, run=run)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
