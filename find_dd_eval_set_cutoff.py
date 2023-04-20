"""Find the cutoff for distances that we will not generate in the distance distortion evaluation set."""

import torch
from typing import List
import config
import pickle
import pathlib
import logging
import argparse
import indices_range
import joblib
import hashlib
import numpy as np
from matplotlib import pyplot as plt


def sample_distance(
    rng: torch.Generator, dim: int, min_value: float, max_value: float
) -> float:
    vec1 = torch.rand(dim, generator=rng) * (max_value - min_value) + min_value
    vec2 = torch.rand(dim, generator=rng) * (max_value - min_value) + min_value
    return torch.norm(vec1 - vec2).item()


def sample_multiple_distances(
    seed: int, dim: int, min_value: float, max_value: float, num_samples: int
) -> List[float]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return [sample_distance(rng, dim, min_value, max_value) for _ in range(num_samples)]


def do_run(
    run: int, rng: np.random.Generator, r_dim: int, n_jobs: int, main_ax: plt.Axes
) -> None:
    min_value = config.MODEL_REPR_DOMAIN[0]
    max_value = config.MODEL_REPR_DOMAIN[1]
    num_samples = config.DDEVSETCUT_NUM_SAMPLES

    # create set of distances between randomly sampled points
    sizes = [num_samples // n_jobs for _ in range(n_jobs)]
    sizes[0] += num_samples - (num_samples // n_jobs) * n_jobs
    sets = joblib.Parallel(n_jobs=n_jobs)(
        [
            joblib.delayed(sample_multiple_distances)(
                int(rng.integers(0, 2**32)),
                r_dim,
                min_value,
                max_value,
                size,
            )
            for size in sizes
        ]
    )
    set = sum(sets, [])

    # find the cutoff point
    sorted_data = np.sort(set)
    cutoff_index = int(config.DDEVSETCUT_CUTOFF * len(sorted_data))
    cutoff = sorted_data[cutoff_index]

    out_file_cutoff = config.DDEVSETCUT_OUT_CUTOFF(run, r_dim)
    pathlib.Path(out_file_cutoff).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_cutoff, "wb") as f:
        pickle.dump(
            cutoff,
            f,
        )

    # plot CDF
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    individual_fig, individual_ax = plt.subplots()
    individual_ax.plot(sorted_data, cdf)
    individual_ax.set_xlabel("Distance")
    individual_ax.set_ylabel("CDF")
    individual_ax.grid()

    out_file_cdfplot = config.DDEVSETCUT_OUT_CDFPLOT_INDIVIDUAL(run, r_dim)
    pathlib.Path(out_file_cdfplot).parent.mkdir(parents=True, exist_ok=True)
    individual_fig.savefig(out_file_cdfplot, bbox_inches="tight")

    plt.close(individual_fig)

    main_ax.plot(sorted_data, cdf, label=f"r_dim={r_dim}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallelism", type=int, default=1)
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
        main_fig, main_ax = plt.subplots()
        main_ax.set_xlabel("Distance")
        main_ax.set_ylabel("CDF")
        main_ax.grid()

        for r_dim_i in args.r_dims:
            r_dim = config.MODEL_R_DIMS[r_dim_i]
            logging.info(f"Finding cutoff for dimension {r_dim}.")

            rng_seed = int(
                hashlib.sha256(
                    f"find_dd_eval_set_cutoff_seed{config.DDEVSETCUT_SEED}_run{run}_r_dim{r_dim}".encode()
                ).hexdigest(),
                16,
            )
            rng = np.random.Generator(np.random.PCG64(rng_seed))

            do_run(
                run=run,
                rng=rng,
                r_dim=r_dim,
                n_jobs=args.parallelism,
                main_ax=main_ax,
            )

        out_file_cdfplot = config.DDEVSETCUT_OUT_CDFPLOT_TOGETHER(run)
        pathlib.Path(out_file_cdfplot).parent.mkdir(parents=True, exist_ok=True)
        main_ax.legend()
        main_fig.savefig(out_file_cdfplot, bbox_inches="tight")


if __name__ == "__main__":
    main()
