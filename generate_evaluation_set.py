"""
For all representations, generate an evaluation set.

This consists of a set of uniformly sampled representations,
and the set of all pairs of representations in this set and their respective distances.
"""

import logging
import config
import torch
import hashlib
from typing import TypeVar, List, Callable, Tuple
from evaluation_set import EvaluationSet
import pathlib
import pickle
import itertools
import argparse
import joblib
import numpy as np

TRepresentation = TypeVar("TRepresentation")


def calculate_multiple_distances(
    pairs: List[Tuple[TRepresentation, TRepresentation]],
    distance: Callable[[TRepresentation, TRepresentation], float],
    slice: Tuple[int, int],
) -> List[float]:
    return [distance(a, b) for (a, b) in pairs[slice[0] : slice[1]]]


def make_evaluation_set(
    representations: List[TRepresentation],
    calc_distance: Callable[[TRepresentation, TRepresentation], float],
    parallelism: int,
) -> EvaluationSet[TRepresentation]:
    pairs = list(itertools.permutations(representations, 2))

    slices = [
        (
            job_i * len(pairs) // parallelism,
            (job_i + 1) * len(pairs) // parallelism,
        )
        for job_i in range(parallelism)
    ]
    slices[-1] = (slices[-1][0], len(pairs))
    results: List[List[float]] = joblib.Parallel(n_jobs=parallelism)(
        [
            joblib.delayed(calculate_multiple_distances)(pairs, calc_distance, slice)
            for slice in slices
        ]
    )
    distances = sum(results, [])

    min_distance = 0.0
    max_distance = max(distances)
    bin_size = (max_distance - min_distance) / config.GENEVAL_NUM_BINS
    bin_ranges = [
        (min_distance + (i) * bin_size, min_distance + (1 + i) * bin_size)
        for i in range(config.GENEVAL_NUM_BINS)
    ]
    bin_indices = np.digitize(distances, [upper for (_, upper) in bin_ranges[0:-1]])

    bins = [[] for _ in range(config.GENEVAL_NUM_BINS)]
    for (repr_1, repr_2), distance, bin_index in zip(pairs, distances, bin_indices):
        bins[bin_index].append((repr_1, repr_2, distance))

    return EvaluationSet(representations, pairs, distances, bins, bin_ranges)


def generate_for_benchmark(run: int) -> None:
    rng_seed = int(
        hashlib.sha256(
            f"generate_evaluation_set_seed{config.GENEVAL_SEED}_benchmark_run{run}".encode()
        ).hexdigest(),
        16,
    )
    raise NotImplementedError()


def generate_for_rtgae(run: int, t_dim: int, r_dim: int, parallelism: int) -> None:
    rng_seed = (
        int(
            hashlib.sha256(
                f"generate_evaluation_set_seed{config.GENEVAL_SEED}_rtgae_run{run}_t_dim{t_dim}_r_dim{r_dim}".encode()
            ).hexdigest(),
            16,
        )
        % 2**64
    )
    rng = torch.Generator()
    rng.manual_seed(rng_seed)

    representations = [
        torch.rand(r_dim, generator=rng)
        * (config.MODEL_REPR_DOMAIN[1] - config.MODEL_REPR_DOMAIN[0])
        + config.MODEL_REPR_DOMAIN[0]
        for _ in range(config.GENEVAL_NUM_REPRESENTATIONS)
    ]

    evaluation_set = make_evaluation_set(
        representations, lambda a, b: torch.norm(a - b).item(), parallelism
    )

    out_file = config.GENEVAL_OUT_RTGAE(run=run, t_dim=t_dim, r_dim=r_dim)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(
            evaluation_set,
            f,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallelism", type=int, default=1)
    args = parser.parse_args()

    for run in range(config.RUNS):
        # logging.info(f"Generating for benchmark run {run}")
        # generate_for_benchmark(run=run, parallelism=args.parallelism) TODO
        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                logging.info(
                    f"Generating for rtgae run {run} t_dim {t_dim} r_dim {r_dim}"
                )
                generate_for_rtgae(
                    run=run, t_dim=t_dim, r_dim=r_dim, parallelism=args.parallelism
                )


if __name__ == "__main__":
    main()
