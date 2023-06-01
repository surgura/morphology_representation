import logging
import pathlib

import matplotlib.pyplot as plt
import pandas
from sqlalchemy import select

import config
import robot_optimization.benchmark.model as bmodel
import robot_optimization.rtgae.model as rmodel
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

            optrun_describes.append(
                describe[["generation_index", "max", "mean", "min"]]
            )

        df = pandas.concat(optrun_describes)
        means = df.groupby(by="generation_index").mean().reset_index()
        means[["max", "mean", "min"]].plot()

        out_dir = config.PLOPT_OUT_MEAN_OPTRUNS_BENCH(run=run)
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir, bbox_inches="tight")

    for run in range(config.RUNS):
        for t_dim_i in range(len(config.MODEL_T_DIMS)):
            for r_dim_i in range(len(config.MODEL_R_DIMS)):
                t_dim = config.MODEL_T_DIMS[t_dim_i]
                r_dim = config.MODEL_R_DIMS[r_dim_i]
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

                    optrun_describes.append(
                        describe[["generation_index", "max", "mean", "min"]]
                    )

                df = pandas.concat(optrun_describes)
                means = df.groupby(by="generation_index").mean().reset_index()
                means[["max", "mean", "min"]].plot()

                out_dir = config.PLOPT_OUT_MEAN_OPTRUNS_RTGAE(run, t_dim, r_dim)
                pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir, bbox_inches="tight")


if __name__ == "__main__":
    main()
