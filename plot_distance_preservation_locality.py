import logging
import pathlib
import pickle

import matplotlib.pyplot as plt

import config
from typing import Tuple, List
import argparse


def plot_pairs(
    pairs: List[Tuple[float, float]], out_file: str, xname: str, yname: str
) -> None:
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="teal")
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_ylim(bottom=config.PLTPREVLOC_Y_LIM[0], top=config.PLTPREVLOC_Y_LIM[1])
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


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
                        with open(
                            config.DPREVRTGAE_OUT(
                                experiment_name=experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                                margin=margin,
                                gain=gain,
                            ),
                            "rb",
                        ) as f:
                            pairs: Tuple[float, float] = pickle.load(f)
                            plot_pairs(
                                pairs,
                                config.PLTPREVLOC_OUT_DPREV(
                                    experiment_name=experiment_name,
                                    run=run,
                                    t_dim=t_dim,
                                    r_dim=r_dim,
                                    margin=margin,
                                    gain=gain,
                                ),
                                "Genotypic distance",
                                "Phenotypic distance",
                            )

                        with open(
                            config.LOCRTGAE_OUT(
                                experiment_name=experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                                margin=margin,
                                gain=gain,
                            ),
                            "rb",
                        ) as f:
                            pairs: Tuple[float, float] = pickle.load(f)
                            plot_pairs(
                                pairs,
                                config.PLTPREVLOC_OUT_LOC(
                                    experiment_name=experiment_name,
                                    run=run,
                                    t_dim=t_dim,
                                    r_dim=r_dim,
                                    margin=margin,
                                    gain=gain,
                                ),
                                "Genotypic distance",
                                "Phenotypic distance",
                            )


if __name__ == "__main__":
    main()
