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
import indices_range
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from tree import DirectedTreeNodeform


def do_run(run: int, parallelism: int, grammar: tree_grammar.TreeGrammar) -> None:
    rng_seed = int(
        hashlib.sha256(
            f"generate_training_set_seed{config.GENTRAIN_RNG_SEED}_run{run}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    archive: List[DirectedTreeNodeform] = [
        DirectedTreeNodeform.random_uniform(10, rng)
        for _ in range(config.GENTRAIN_ARCHIVE_SIZE)
    ]
    out_file = config.GENTRAIN_OUT(run)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(archive, f)


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
    args = parser.parse_args()

    grammar = make_body_rgt()

    for run in args.runs:
        do_run(run=run, parallelism=args.parallelism, grammar=grammar)


if __name__ == "__main__":
    main()
