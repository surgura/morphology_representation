"""Make sets of pairs of vectors with a uniform PDF for distance between them, uniformly sampled throughout the vector hypercube."""

import torch
import math
from typing import Tuple, List
import config
import pickle
import pathlib
import logging
import argparse
import indices_range
import joblib
import hashlib
import numpy as np


def make_random_vector(
    rng: torch.Generator, dim: int, min_value: float, max_value: float
) -> torch.Tensor:
    return torch.rand(dim, generator=rng) * (max_value - min_value) + min_value


def is_hypercube_within_hypersphere(
    hypercube_center: torch.Tensor,
    hypercube_side_length: float,
    hypersphere_center: torch.Tensor,
    hypersphere_radius: float,
) -> bool:
    # calculate distance between center of hypercube and center of hypersphere
    distance = torch.norm(hypercube_center - hypersphere_center)

    # calculate diagonal length of hypercube
    n = hypercube_center.shape[0]  # number of dimensions
    diagonal_length = hypercube_side_length * torch.sqrt(torch.tensor(n))

    # check if hypercube is completely within hypersphere
    return diagonal_length / 2 <= hypersphere_radius - distance


def make_vector_pair(
    rng: torch.Generator, dim: int, min_value: float, max_value: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    hypercube_side_length = max_value - min_value

    while True:
        max_distance = math.sqrt(dim * hypercube_side_length**2)
        desired_distance = (torch.rand(1, generator=rng) * max_distance).item()

        # generate a vector such that there exists at least one other vector that is desired_distance away
        failcounter = 0
        vec1: torch.Tensor
        while failcounter != config.DDEVSET_MAX_FAILS:
            vec1 = make_random_vector(rng, dim, min_value, max_value)
            if not is_hypercube_within_hypersphere(
                hypercube_center=torch.tensor([0.0] * dim),
                hypercube_side_length=hypercube_side_length,
                hypersphere_center=vec1,
                hypersphere_radius=desired_distance,
            ):
                break
            failcounter += 1

        if failcounter >= config.DDEVSET_MAX_FAILS:
            continue

        # generate a second vector that is exactly desired_distance away from the first vector
        failcounter = 0
        vec2: torch.Tensor
        while failcounter != config.DDEVSET_MAX_FAILS:
            direction = make_random_vector(rng, dim, min_value, max_value)
            direction /= torch.norm(direction)
            vec2 = vec1 + direction * desired_distance
            if torch.all(vec2 >= min_value) and torch.all(vec2 <= max_value):
                return vec1, vec2
            failcounter += 1


def make_set(
    seed: int, dim: int, min_value: float, max_value: float, num_pairs: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    rng = torch.Generator()
    print(seed)
    rng.manual_seed(seed)
    return [
        make_vector_pair(rng=rng, dim=dim, min_value=min_value, max_value=max_value)
        for _ in range(num_pairs)
    ]


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
        for r_dim_i in args.r_dims:
            r_dim = config.MODEL_R_DIMS[r_dim_i]
            logging.info(f"Making set for dimension {r_dim}.")

            rng_seed = int(
                hashlib.sha256(
                    f"make_dd_eval_set_seed{config.DDEVSET_SEED}_run{run}_r_dim{r_dim}".encode()
                ).hexdigest(),
                16,
            )
            rng = np.random.Generator(np.random.PCG64(rng_seed))

            n_jobs = args.parallelism
            sizes = [config.DDEVSET_NUM_PAIRS // n_jobs for _ in range(n_jobs)]
            sizes[0] += (
                config.DDEVSET_NUM_PAIRS - (config.DDEVSET_NUM_PAIRS // n_jobs) * n_jobs
            )
            results = joblib.Parallel(n_jobs=n_jobs)(
                [
                    joblib.delayed(make_set)(
                        int(rng.integers(0, 2**32)),
                        r_dim,
                        config.MODEL_REPR_DOMAIN[0],
                        config.MODEL_REPR_DOMAIN[1],
                        size,
                    )
                    for size in sizes
                ]
            )
            results_concat = sum(results, [])

            print(results_concat)

            out_file = config.DDEVSET_OUT(run, r_dim)
            pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, "wb") as f:
                pickle.dump(
                    results_concat,
                    f,
                )


if __name__ == "__main__":
    main()

# def plot_pdf(dim, n_samples=10000, n_bins=50):
#     import matplotlib.pyplot as plt
#     distances = []

#     for i in range(n_samples):
#         print(i)
#         vector_a, vector_b = uniform_distance_vectors(dim, min_value=-1, max_value=1)
#         distance = torch.norm(vector_a - vector_b).numpy()
#         distances.append(distance)

#     print("done")

#     plt.hist(
#         distances, bins=n_bins, density=True, alpha=0.75, label="Euclidean distance PDF"
#     )
#     plt.xlabel("Distance")
#     plt.ylabel("Probability Density")
#     plt.title(f"PDF of Euclidean distances between random vectors (dim={dim})")
#     plt.legend()
#     plt.show()


# # Example usage:
# dim = 100
# plot_pdf(dim)
