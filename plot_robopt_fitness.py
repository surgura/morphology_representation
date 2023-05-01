import logging
import pathlib

import matplotlib.pyplot as plt
import pandas
from sqlalchemy import select

import config
import robot_optimization.benchmark.model as bmodel
import robot_optimization.rtgae.model as rmodel
from revolve2.core.database import open_database_sqlite


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    for run in range(config.RUNS):
        optrun_describes = []
        for optrun in range(config.ROBOPT_RUNS):
            db = open_database_sqlite(config.OPTBENCH_OUT(run, optrun))
            df = pandas.read_sql(
                select(bmodel.Generation.generation_index, bmodel.Individual.fitness)
                .join(bmodel.Generation.population)
                .join(bmodel.Population.individuals),
                db,
            )
            describe = (
                df.groupby(by="generation_index").describe()["fitness"].reset_index()
            )
            describe[["max", "mean", "min"]].plot()

            out_dir = config.PLOPT_OUT_INDIVIDUAL_OPTRUNS_BENCH(run, optrun)
            pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir, bbox_inches="tight")

            optrun_describes.append(
                describe[["generation_index", "max", "mean", "min"]]
            )

        df = pandas.concat(optrun_describes)
        means = df.groupby(by="generation_index").mean().reset_index()
        means[["max", "mean", "min"]].plot()

        out_dir = config.PLOPT_OUT_MEAN_OPTRUNS_BENCH(run)
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir, bbox_inches="tight")

    for run in range(config.RUNS):
        for bestorworst in [True, False]:
            optrun_describes = []
            for optrun in range(config.ROBOPT_RUNS):
                db = open_database_sqlite(config.OPTRTGAE_OUT(run, optrun, bestorworst))
                df = pandas.read_sql(
                    select(
                        rmodel.Generation.generation_index, rmodel.Individual.fitness
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
                    run, optrun, bestorworst
                )
                pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir, bbox_inches="tight")

                optrun_describes.append(
                    describe[["generation_index", "max", "mean", "min"]]
                )

            df = pandas.concat(optrun_describes)
            means = df.groupby(by="generation_index").mean().reset_index()
            means[["max", "mean", "min"]].plot()

            out_dir = config.PLOPT_OUT_MEAN_OPTRUNS_RTGAE(run, bestorworst)
            pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir, bbox_inches="tight")


if __name__ == "__main__":
    main()
