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
                for margin in config.TRAIN_DD_MARGINS:
                    for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                        in_dir = config.TRAIN_DD_OUT_LOSS(
                            experiment_name=experiment_name,
                            run=run,
                            t_dim=t_dim,
                            r_dim=r_dim,
                            margin=margin,
                            gain=gain,
                        )
                        with open(in_dir, "rb") as f:
                            all_losses = pickle.load(f)
                        losses = all_losses["losses"]
                        recon_losses = all_losses["recon_losses"]
                        metric_losses = all_losses["metric_losses"]

                        df = pandas.DataFrame.from_records(
                            zip(
                                [i for i in range(len(losses))],
                                losses,
                                recon_losses,
                                metric_losses,
                            ),
                            columns=("epoch", "loss", "recon_loss", "metric_loss"),
                        )
                        ax = df.plot(x="epoch", y="loss", legend=False)
                        # df.plot(ax=ax, x="epoch", y="recon_loss", legend=False)
                        # df.plot(ax=ax, x="epoch", y="metric_loss", legend=False)
                        ax.set_ylabel("loss")

                        out_dir = config.PLTTRAIN_OUT(
                            experiment_name=experiment_name,
                            run=run,
                            t_dim=t_dim,
                            r_dim=r_dim,
                            margin=margin,
                            gain=gain,
                        )
                        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(out_dir)
                        plt.close()

        ax = plt.subplot()
        ax.set_ylabel("loss")
        # for t_dim in config.MODEL_T_DIMS:
        #     for r_dim in config.MODEL_R_DIMS:
        for t_dim, r_dim in [
            (64, 8),
            (64, 16),
            (128, 8),
            (256, 8),
            (64, 48),
            # (256, 24),
            (128, 24),
            (128, 48),
        ]:
            for margin in config.TRAIN_DD_MARGINS:
                for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                    in_dir = config.TRAIN_DD_OUT_LOSS(
                        experiment_name=experiment_name,
                        run=run,
                        t_dim=t_dim,
                        r_dim=r_dim,
                        margin=margin,
                        gain=gain,
                    )
                    with open(in_dir, "rb") as f:
                        all_losses = pickle.load(f)
                    losses = all_losses["losses"]
                    recon_losses = all_losses["recon_losses"]
                    metric_losses = all_losses["metric_losses"]

                    df = pandas.DataFrame.from_records(
                        zip(
                            [i for i in range(len(losses))],
                            losses,
                            recon_losses,
                            metric_losses,
                        ),
                        columns=("epoch", "loss", "recon_loss", "metric_loss"),
                    )
                    df.plot(
                        ax=ax,
                        x="epoch",
                        y="loss",
                        label=f"tree={t_dim} vae={r_dim}",
                    )

                    # df.plot(ax=ax, x="epoch", y="recon_loss", legend=False)
                    # ax.text(
                    #     df["epoch"].iloc[-1],
                    #     df["recon_loss"].iloc[-1],
                    #     f"recon_loss t={t_dim} r={r_dim} margin={margin} gain={gain}",
                    #     fontsize=9,
                    #     va="center",
                    #     ha="left",
                    # )

                    # df.plot(ax=ax, x="epoch", y="metric_loss", legend=False)
                    # ax.text(
                    #     df["epoch"].iloc[-1],
                    #     df["metric_loss"].iloc[-1],
                    #     f"metric_loss t={t_dim} r={r_dim} margin={margin} gain={gain}",
                    #     fontsize=9,
                    #     va="center",
                    #     ha="left",
                    # )
        plt.ylabel("Training loss (logarithmic)")
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.legend(loc="upper right")
        out_dir = config.PLTTRAIN_OUT_ALL(experiment_name=experiment_name, run=run)
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir, bbox_inches="tight")
        plt.close()
        plt.close()


if __name__ == "__main__":
    main()
