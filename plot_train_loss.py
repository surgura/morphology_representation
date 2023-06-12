import argparse
import logging
import pathlib
import pickle
from typing import Tuple

import joblib
import config
import indices_range
import matplotlib.pyplot as plt
import pandas


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
        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                in_dir = config.TRAIN_OUT_LOSS(
                    experiment_name=experiment_name,
                    run=run,
                    t_dim=t_dim,
                    r_dim=r_dim,
                )
                with open(in_dir, "rb") as f:
                    losses = pickle.load(f)

                df = pandas.DataFrame.from_records(
                    zip([i for i in range(len(losses))], losses),
                    columns=("epoch", "loss"),
                )
                ax = df.plot(x="epoch", y="loss", legend=False)
                ax.set_ylabel("loss")

                out_dir = config.PLTTRAIN_OUT(
                    experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
                )
                pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir)
                plt.close()

        ax = plt.subplot()
        ax.set_ylabel("loss")
        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                in_dir = config.TRAIN_OUT_LOSS(
                    experiment_name=experiment_name,
                    run=run,
                    t_dim=t_dim,
                    r_dim=r_dim,
                )
                with open(in_dir, "rb") as f:
                    losses = pickle.load(f)

                df = pandas.DataFrame.from_records(
                    zip([i for i in range(len(losses))], losses),
                    columns=("epoch", "loss"),
                )
                ax = df.plot(x="epoch", y="loss", ax=ax, label=f"t={t_dim} r={r_dim}")
                ax.text(
                    df["epoch"].iloc[-1],
                    df["loss"].iloc[-1],
                    f"t={t_dim} r={r_dim}",
                    fontsize=9,
                    va="center",
                    ha="left",
                )
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
