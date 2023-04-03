import logging
import config
import pandas
import robot_optimization.benchmark.model as bmodel
import robot_optimization.rtgae.model as rmodel
from sqlalchemy import select
from revolve2.core.database import open_database_sqlite
import matplotlib.pyplot as plt
import pathlib


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    for run in range(config.RUNS):
        for optrun in range(config.ROBOPT_RUNS):
            db = open_database_sqlite(config.OPTBENCH_OUT(run, optrun))
            df = pandas.read_sql(
                select(bmodel.Generation, bmodel.Population, bmodel.Individual)
                .join(bmodel.Generation.population)
                .join(bmodel.Population.individuals),
                db,
            )
            describe = (
                df[["generation_index", "fitness"]]
                .groupby(by="generation_index")
                .describe()["fitness"]
            )
            describe[["max", "mean", "min"]].plot()

            out_dir = config.PLOPT_OUT_INDIVIDUAL_OPTRUNS_BENCH(run, optrun)
            pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir)

            for bestorworst in [True, False]:
                db = open_database_sqlite(config.OPTRTGAE_OUT(run, optrun, bestorworst))
                df = pandas.read_sql(
                    select(rmodel.Generation, rmodel.Population, rmodel.Individual)
                    .join(rmodel.Generation.population)
                    .join(rmodel.Population.individuals),
                    db,
                )
                describe = (
                    df[["generation_index", "fitness"]]
                    .groupby(by="generation_index")
                    .describe()["fitness"]
                )
                describe[["max", "mean", "min"]].plot()

                out_dir = config.PLOPT_OUT_INDIVIDUAL_OPTRUNS_RTGAE(
                    run, optrun, bestorworst
                )
                pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir)


if __name__ == "__main__":
    main()
