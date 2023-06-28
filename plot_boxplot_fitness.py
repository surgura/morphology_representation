import logging
import pathlib

import matplotlib.pyplot as plt
import pandas
from sqlalchemy import select, func

import config
import robot_optimization.benchmark.model as bmodel
import robot_optimization.rtgae.model as rmodel
import robot_optimization.cmaes.model as cmodel
from revolve2.core.database import open_database_sqlite
import argparse
from sqlalchemy.orm import Session
from typing import List, Dict


def plot_with_error_bars(df, y_col, label, color, ax):
    aggregated = df.groupby("performed_evaluations").agg(["mean", "std"])
    means = aggregated[y_col]["mean"]
    stds = aggregated[y_col]["std"]

    means.plot(y=y_col, ax=ax, color=color, label=label)
    ax.fill_between(means.index, means - stds, means + stds, color=color, alpha=0.2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    args = parser.parse_args()

    excmaes = "paper_nometric"
    excppn = "papercppn"

    for run in range(config.RUNS):
        ax = plt.subplot()

        best_fitnesses_cppn: List[float] = []
        best_fitnesses_cmaes: Dict[str, List[float]] = {}

        # CPPN
        for optrun in range(config.ROBOPT_RUNS):
            db = open_database_sqlite(
                config.OPTBENCH_OUT(experiment_name=excppn, run=run, optrun=optrun)
            )
            with Session(db) as ses:
                best_fitness = ses.scalar(select(func.max(bmodel.Individual.fitness)))
            best_fitnesses_cppn.append(best_fitness)

        # CMAES
        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                for margin in config.TRAIN_DD_MARGINS:
                    for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                        idx = f"{t_dim}_{r_dim}_{margin}_{gain}"
                        best_fitnesses_cmaes[idx] = []
                        for optrun in range(config.ROBOPT_RUNS):
                            db = open_database_sqlite(
                                config.OPTCMAES_OUT(
                                    experiment_name=excmaes,
                                    run=run,
                                    optrun=optrun,
                                    t_dim=t_dim,
                                    r_dim=r_dim,
                                    margin=margin,
                                    gain=gain,
                                )
                            )
                            with Session(db) as ses:
                                best_fitness = ses.scalar(
                                    select(func.max(cmodel.Generation.fitness))
                                )
                            best_fitnesses_cmaes[idx].append(best_fitness)

        ax = plt.subplot()
        ax.boxplot(
            [best_fitnesses_cppn] + list(best_fitnesses_cmaes.values()),
            boxprops={"color": "teal", "linewidth": 1.1},
            whiskerprops={"color": "teal", "linewidth": 1.1},
            capprops={"color": "teal", "linewidth": 1.1},
            medianprops={"color": "sandybrown", "linewidth": 1.1},
            widths=0.5,
        )
        ax.set_xticklabels(["CPPN", "Vector"])
        plt.ylabel("Fitness")

        out_dir = config.BXPLT_OUT(
            experiment_name=args.experiment_name,
            run=run,
        )
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        ax.set_aspect(0.5)
        plt.savefig(out_dir, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
