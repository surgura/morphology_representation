import matplotlib.pyplot as plt
import pickle
import config
from typing import List, Tuple
import torch
import logging
import argparse
import indices_range


def plot_pdf(run: int, r_dim: int):
    distances = []

    file = config.DDEVSET_OUT(run, r_dim)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    with open(file, "rb") as f:
        pairs = pickle.load(f)
    distances: float = [torch.norm(pair[0] - pair[1]).item() for pair in pairs]

    plt.hist(
        distances,
        bins=config.DDEVSETPLOT_NUM_BINS,
        density=True,
        alpha=0.75,
        label="Euclidean distance PDF",
    )
    plt.xlabel("Distance")
    plt.ylabel("Probability Density")
    plt.title(
        f"PDF of Euclidean distances between random vectors (run={run}, r_dim={r_dim})"
    )
    plt.legend()
    plt.show()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runs",
        type=indices_range.indices_type(range(config.RUNS)),
        required=True,
    )
    parser.add_argument(
        "--r_dims",
        type=indices_range.indices_type(range(len(config.MODEL_R_DIMS))),
        required=True,
    )
    args = parser.parse_args()

    for run in args.runs:
        for r_dim_i in args.r_dims:
            r_dim = config.MODEL_R_DIMS[r_dim_i]
            plot_pdf(run=run, r_dim=r_dim)


if __name__ == "__main__":
    main()
