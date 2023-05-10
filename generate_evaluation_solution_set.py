"""
Find a set of trees that reasonably represent the complete space of robots using novelty and mutation.
"""

import argparse
import hashlib
import logging
import pathlib
import pickle
from typing import List

import numpy as np

import config
from tree import DirectedTreeNodeform


def do_run(run: int) -> None:
    rng_seed = int(
        hashlib.sha256(
            f"generate_evaluation_solution_set_seed{config.GENEVALSOL_RNG_SEED}_run{run}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    archive: List[DirectedTreeNodeform] = sum(
        [
            [
                DirectedTreeNodeform.random_uniform(size, rng)
                for _ in range(
                    config.GENEVALSOL_ARCHIVE_SIZE // (config.MODEL_MAX_MODULES + 1 - 1)
                )
            ]
            for size in range(1, config.MODEL_MAX_MODULES + 1)
        ],
        [],
    )
    out_file = config.GENEVALSOL_OUT(run)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(archive, f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.parse_args()

    for run in range(config.RUNS):
        do_run(run=run)


if __name__ == "__main__":
    main()
