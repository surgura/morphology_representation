import logging
import pathlib

import matplotlib.pyplot as plt
import pandas
from sqlalchemy import select

import config
import robot_optimization.benchmark.model as bmodel
import robot_optimization.rtgae.model as rmodel
import robot_optimization.cmaes.model as cmodel
from revolve2.core.database import open_database_sqlite
import argparse


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

    experiment_name = args.experiment_name

    cppn = False

    for run in range(config.RUNS):
        # CPPN
        if cppn:
            dfs = []
            for optrun in range(config.ROBOPT_RUNS):
                db = open_database_sqlite(
                    config.OPTBENCH_OUT(
                        experiment_name=experiment_name, run=run, optrun=optrun
                    )
                )
                df = pandas.read_sql(
                    select(
                        bmodel.Generation.generation_index,
                        bmodel.Individual.fitness,
                        bmodel.Individual.fitness_before_learning,
                    )
                    .join(bmodel.Generation.population)
                    .join(bmodel.Population.individuals),
                    db,
                )

                df["performed_evaluations"] = (
                    df["generation_index"]
                    * config.OPTCMAES_NUM_EVALUATIONS
                    // config.ROBOPT_NUM_GENERATIONS
                )
                df["learning_delta"] = df["fitness"] - df["fitness_before_learning"]

                grouped = df.groupby("performed_evaluations").apply(
                    lambda x: x.sort_values(
                        ["fitness", "learning_delta"], ascending=[False, False]
                    )
                )
                grouped.reset_index(drop=True, inplace=True)
                best = grouped.groupby("performed_evaluations").first()
                best["cummax"] = best["fitness"].cummax()
                best["cummax_learning_delta"] = [
                    best.iloc[: i + 1]["fitness"].idxmax() for i in range(len(best))
                ]
                best["cummax_learning_delta"] = best["cummax_learning_delta"].map(
                    best["learning_delta"]
                )
                best.reset_index(inplace=True)

                meanmax = df.groupby("performed_evaluations")[
                    ["fitness", "learning_delta"]
                ].agg(["mean", "max"])
                meanmax.columns = [
                    "_".join(col).strip() for col in meanmax.columns.values
                ]
                meanmax.reset_index(inplace=True)

                df = pandas.concat(
                    [
                        best[
                            ["performed_evaluations", "cummax", "cummax_learning_delta"]
                        ].set_index("performed_evaluations"),
                        meanmax.set_index("performed_evaluations"),
                    ],
                    axis=1,
                ).reset_index()

                dfs.append(df)

            df = pandas.concat(dfs)

            # fitness plot
            ax = plt.subplot()

            plot_with_error_bars(
                df[["performed_evaluations", "cummax"]], "cummax", "Best", "teal", ax
            )
            plot_with_error_bars(
                df[["performed_evaluations", "fitness_mean"]],
                "fitness_mean",
                "Mean",
                "sandybrown",
                ax,
            )

            ax.set_ylabel("Fitness")
            ax.set_xlabel("Evaluations")
            ax.set_ylim(bottom=0.0, top=config.PLOPT_FITNESS_Y)
            plt.legend(loc="upper left")

            out_dir = config.PLOPT_OUT_FITNESS_OPTRUNS_BENCH(
                experiment_name=experiment_name, run=run
            )
            pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir, bbox_inches="tight")
            plt.close()

            # learning delta plot
            ax = plt.subplot()

            plot_with_error_bars(
                df[["performed_evaluations", "cummax_learning_delta"]],
                "cummax_learning_delta",
                "Best",
                "teal",
                ax,
            )
            plot_with_error_bars(
                df[["performed_evaluations", "learning_delta_mean"]],
                "learning_delta_mean",
                "Mean",
                "sandybrown",
                ax,
            )

            ax.set_ylabel("Learning delta")
            ax.set_xlabel("Evaluations")
            ax.set_ylim(bottom=0.0, top=config.PLOPT_LEARNINGDELTA_Y)
            plt.legend(loc="upper left")

            out_dir = config.PLOPT_OUT_LEARNINGDELTA_OPTRUNS_BENCH(
                experiment_name=experiment_name, run=run
            )
            pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir, bbox_inches="tight")
            plt.close()
        else:
            # CMAES
            for t_dim in config.MODEL_T_DIMS:
                for r_dim in config.MODEL_R_DIMS:
                    for margin in config.TRAIN_DD_MARGINS:
                        for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                            dfs = []
                            for optrun in range(config.ROBOPT_RUNS):
                                db = open_database_sqlite(
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
                                df = pandas.read_sql(
                                    select(
                                        cmodel.Generation.performed_evaluations,
                                        cmodel.Generation.fitness,
                                        cmodel.Generation.fitness_before_learning,
                                        cmodel.PopIndividual.fitness.label(
                                            "popin_fitness"
                                        ),
                                        cmodel.PopIndividual.fitness_before_learning.label(
                                            "popin_fitness_before_learning"
                                        ),
                                    )
                                    .join_from(cmodel.Generation, cmodel.SamplePop)
                                    .join_from(cmodel.SamplePop, cmodel.PopIndividual),
                                    db,
                                )

                                df["cummax_learning_delta"] = (
                                    df["fitness"] - df["fitness_before_learning"]
                                )
                                df["popin_learning_delta"] = (
                                    df["popin_fitness"]
                                    - df["popin_fitness_before_learning"]
                                )

                                best = df.groupby("performed_evaluations").first()[
                                    ["fitness", "cummax_learning_delta"]
                                ]
                                best.rename(columns={"fitness": "cummax"}, inplace=True)
                                best.reset_index(inplace=True)

                                meanmax = df.groupby("performed_evaluations")[
                                    ["popin_fitness", "popin_learning_delta"]
                                ].agg(["mean", "max"])
                                meanmax.columns = [
                                    "_".join(col).strip()
                                    for col in meanmax.columns.values
                                ]
                                meanmax.reset_index(inplace=True)

                                df = pandas.concat(
                                    [
                                        best[
                                            [
                                                "performed_evaluations",
                                                "cummax",
                                                "cummax_learning_delta",
                                            ]
                                        ].set_index("performed_evaluations"),
                                        meanmax.set_index("performed_evaluations"),
                                    ],
                                    axis=1,
                                ).reset_index()

                                dfs.append(df)

                            df = pandas.concat(dfs)

                            # fitness plot
                            ax = plt.subplot()

                            plot_with_error_bars(
                                df[["performed_evaluations", "cummax"]],
                                "cummax",
                                "Best",
                                "teal",
                                ax,
                            )
                            plot_with_error_bars(
                                df[["performed_evaluations", "popin_fitness_mean"]],
                                "popin_fitness_mean",
                                "Mean",
                                "sandybrown",
                                ax,
                            )

                            ax.set_ylabel("Fitness")
                            ax.set_xlabel("Evaluations")
                            ax.set_ylim(bottom=0.0, top=config.PLOPT_FITNESS_Y)
                            plt.legend(loc="upper left")

                            out_dir = config.PLOPT_OUT_FITNESS_OPTRUNS_CMAES(
                                experiment_name=experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                            )
                            pathlib.Path(out_dir).parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            plt.savefig(out_dir, bbox_inches="tight")
                            plt.close()

                            # learning delta plot
                            ax = plt.subplot()

                            plot_with_error_bars(
                                df[["performed_evaluations", "cummax_learning_delta"]],
                                "cummax_learning_delta",
                                "Best",
                                "teal",
                                ax,
                            )
                            plot_with_error_bars(
                                df[
                                    [
                                        "performed_evaluations",
                                        "popin_learning_delta_mean",
                                    ]
                                ],
                                "popin_learning_delta_mean",
                                "Mean",
                                "sandybrown",
                                ax,
                            )

                            ax.set_ylabel("Learning delta")
                            ax.set_xlabel("Evaluations")
                            ax.set_ylim(bottom=0.0, top=config.PLOPT_LEARNINGDELTA_Y)
                            plt.legend(loc="upper left")

                            out_dir = config.PLOPT_OUT_LEARNINGDELTA_OPTRUNS_CMAES(
                                experiment_name=experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                            )
                            pathlib.Path(out_dir).parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            plt.savefig(out_dir, bbox_inches="tight")
                            plt.close()


if __name__ == "__main__":
    main()
