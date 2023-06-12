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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    args = parser.parse_args()

    experiment_name = args.experiment_name

    for run in range(config.RUNS):
        # CPPN
        optrun_describes = []
        for optrun in range(config.ROBOPT_RUNS):
            db = open_database_sqlite(
                config.OPTBENCH_OUT(
                    experiment_name=experiment_name, run=run, optrun=optrun
                )
            )
            df = pandas.read_sql(
                select(bmodel.Generation.generation_index, bmodel.Individual.fitness)
                .join(bmodel.Generation.population)
                .join(bmodel.Population.individuals),
                db,
            )

            describe = (
                df.groupby(by="generation_index").describe()["fitness"].reset_index()
            )
            describe.set_index("generation_index")[["max", "mean", "min"]].plot()

            out_dir = config.PLOPT_OUT_INDIVIDUAL_OPTRUNS_BENCH(
                experiment_name=experiment_name, run=run, optrun=optrun
            )
            pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir, bbox_inches="tight")
            plt.close()

            optrun_describes.append(
                describe[["generation_index", "max", "mean", "min"]]
            )

        df = pandas.concat(optrun_describes)
        mean = df.groupby(by="generation_index").mean()
        mean.plot()
        cppn = mean["max"].reset_index()
        cppn["generation_index"] *= (
            config.OPTCMAES_NUM_EVALUATIONS / config.ROBOPT_NUM_GENERATIONS
        )
        cppn.rename(columns={"generation_index": "evaluations"}, inplace=True)

        out_dir = config.PLOPT_OUT_MEAN_OPTRUNS_BENCH(
            experiment_name=experiment_name, run=run
        )
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir, bbox_inches="tight")
        plt.close()

        # RTGAE
        rtgae = {}

        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                optrun_describes = []
                for optrun in range(config.ROBOPT_RUNS):
                    db = open_database_sqlite(
                        config.OPTRTGAE_OUT(experiment_name, run, optrun, t_dim, r_dim)
                    )
                    df = pandas.read_sql(
                        select(
                            rmodel.Generation.generation_index,
                            rmodel.Individual.fitness,
                        )
                        .join(rmodel.Generation.population)
                        .join(rmodel.Population.individuals),
                        db,
                    )
                    describe = (
                        df.groupby(by="generation_index")
                        .describe()["fitness"]
                        .reset_index()
                    )
                    describe[["max", "mean", "min"]].plot()

                    out_dir = config.PLOPT_OUT_INDIVIDUAL_OPTRUNS_RTGAE(
                        experiment_name, run, optrun, t_dim, r_dim
                    )
                    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(out_dir, bbox_inches="tight")
                    plt.close()

                    optrun_describes.append(
                        describe[["generation_index", "max", "mean", "min"]]
                    )

                df = pandas.concat(optrun_describes)
                mean = df.groupby(by="generation_index").mean()
                mean.plot()
                rtgae[f"t_{t_dim}___r_{r_dim}"] = mean["max"].reset_index()
                rtgae[f"t_{t_dim}___r_{r_dim}"]["generation_index"] *= (
                    config.OPTCMAES_NUM_EVALUATIONS / config.ROBOPT_NUM_GENERATIONS
                )
                rtgae[f"t_{t_dim}___r_{r_dim}"].rename(
                    columns={"generation_index": "evaluations"}, inplace=True
                )

                out_dir = config.PLOPT_OUT_MEAN_OPTRUNS_RTGAE(
                    experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
                )
                pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir, bbox_inches="tight")
                plt.close()

        # CMAES
        cmaes = {}

        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                dfs = []
                for optrun in range(config.ROBOPT_RUNS):
                    db = open_database_sqlite(
                        config.OPTCMAES_OUT(experiment_name, run, optrun, t_dim, r_dim)
                    )
                    df = pandas.read_sql(
                        select(
                            cmodel.Generation.generation_index,
                            cmodel.Generation.fitness,
                        ),
                        db,
                    )
                    df.set_index("generation_index").plot()

                    out_dir = config.PLOPT_OUT_INDIVIDUAL_OPTRUNS_CMAES(
                        experiment_name, run, optrun, t_dim, r_dim
                    )
                    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(out_dir, bbox_inches="tight")
                    plt.close()

                    dfs.append(df)

                df = pandas.concat(dfs)
                mean = df.groupby(by="generation_index").mean()
                mean.plot()
                cmaes[f"t_{t_dim}___r_{r_dim}"] = mean.reset_index().rename(
                    columns={"fitness": "max", "generation_index": "evaluations"}
                )
                cmaes[f"t_{t_dim}___r_{r_dim}"]["evaluations"] *= (
                    config.OPTCMAES_NUM_EVALUATIONS
                    / cmaes[f"t_{t_dim}___r_{r_dim}"]["evaluations"].max()
                )

                out_dir = config.PLOPT_OUT_MEAN_OPTRUNS_CMAES(
                    experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
                )
                pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir, bbox_inches="tight")
                plt.close()

        # ax = cppn.plot(x="generation_index")
        out_dir = config.PLOPT_OUT_ALL(experiment_name=experiment_name, run=run)
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        ax = cppn.plot(x="evaluations", y="max", label="CPPNWIN")
        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                rtgae[f"t_{t_dim}___r_{r_dim}"].plot(
                    ax=ax, x="evaluations", y="max", label=f"RTGAE t={t_dim} r={r_dim}"
                )
                cmaes[f"t_{t_dim}___r_{r_dim}"].plot(
                    ax=ax, x="evaluations", y="max", label=f"CMAES t={t_dim} r={r_dim}"
                )

        ax.set_ylabel("Fitness")
        ax.set_xlabel("Evaluations")
        plt.savefig(out_dir, bbox_inches="tight")


if __name__ == "__main__":
    main()
