"""
Find a set of trees that reasonably represent the complete space of robots using novelty and mutation.
"""

from typing import List, Tuple, Dict, FrozenSet, Set
from tree import DirectedTreeNodeform
import numpy as np
import config
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from pqgrams_util import tree_to_pqgrams
from pqgrams import Profile as PqgramsProfile  # type: ignore
import joblib
import pickle
import indices_range
import logging
import argparse
import hashlib

import random
import numpy as np
import pathlib


def make_random_tree(
    rng: np.random.Generator, grammar: tree_grammar.TreeGrammar
) -> DirectedTreeNodeform:
    tree = DirectedTreeNodeform()
    for _ in range(config.GENTRAIN_INITIAL_MUTATIONS):
        tree.mutate_binomial(
            rng,
            grammar,
            max_modules=config.MODEL_MAX_MODULES,
            n=config.GENTRAIN_MUTATE_N,
            p=config.GENTRAIN_MUTATE_P,
        )
    return tree


def make_initial_population(
    rng: np.random.Generator, grammar: tree_grammar.TreeGrammar
) -> List[DirectedTreeNodeform]:
    return [
        make_random_tree(rng, grammar) for _ in range(config.GENTRAIN_POPULATION_SIZE)
    ]


def measure_distances_parallel(
    population: List[Tuple[DirectedTreeNodeform, DirectedTreeNodeform]],
    slice: Tuple[int, int],
) -> List[List[float]]:
    return [
        tree_to_pqgrams(tree1.to_graph_adjform()).edit_distance(
            tree_to_pqgrams(tree2.to_graph_adjform())
        )
        for tree1, tree2 in population[slice[0] : slice[1]]
    ]


def measure_distances(
    to_measure: List[Tuple[DirectedTreeNodeform, DirectedTreeNodeform]], num_jobs: int
) -> List[float]:
    # print(len(to_measure))
    slices = [
        (
            job_i * len(to_measure) // num_jobs,
            (job_i + 1) * len(to_measure) // num_jobs,
        )
        for job_i in range(num_jobs)
    ]
    slices[-1] = (slices[-1][0], len(to_measure))
    results = joblib.Parallel(n_jobs=num_jobs)(
        [
            joblib.delayed(measure_distances_parallel)(
                to_measure,
                slice,
            )
            for slice in slices
        ]
    )
    return sum(results, [])


class NoveltyDatabase:
    items: Set[DirectedTreeNodeform] = set()
    comparisons: Dict[FrozenSet[DirectedTreeNodeform], float] = {}
    unknown: Set[FrozenSet[DirectedTreeNodeform]] = set()

    def add_item(self, newitem: DirectedTreeNodeform) -> None:
        assert newitem not in self.items
        for item in self.items:
            self.unknown.add(frozenset((item, newitem)))
        self.items.add(newitem)

    def remove_item(self, olditem: DirectedTreeNodeform) -> None:
        assert len(self.unknown) == 0
        self.items.remove(olditem)
        for item in self.items:
            self.comparisons.pop(frozenset((olditem, item)))

    def novelty(self, item: DirectedTreeNodeform) -> float:
        assert len(self.unknown) == 0
        return self.__knnavg(
            [
                self.comparisons[frozenset((item, other))]
                for other in self.items
                if item is not other
            ]
        )

    def get_unknown_differences(
        self,
    ) -> List[FrozenSet[DirectedTreeNodeform]]:
        return list(self.unknown)

    def set_unknown_differences(
        self,
        differences: List[Tuple[FrozenSet[DirectedTreeNodeform], float]],
    ):
        for pair, diff in differences:
            self.comparisons[pair] = diff
            self.unknown.remove(pair)

    @staticmethod
    def __knnavg(diffs: List[float]) -> float:
        diffs.sort()
        return sum(diffs[-config.GENTRAIN_KNN_K :])

    def novelty_subset(self, items: List[DirectedTreeNodeform]) -> List[float]:
        return [
            self.__knnavg(
                [
                    self.comparisons[frozenset((item, other))]
                    for other in items
                    if item is not other
                ]
            )
            for item in items
        ]


def do_run(run: int, parallelism: int, grammar: tree_grammar.TreeGrammar) -> None:
    rng_seed = int(
        hashlib.sha256(
            f"generate_training_set_seed{config.GENTRAIN_RNG_SEED}_run{run}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    nvdb = NoveltyDatabase()
    archive: List[DirectedTreeNodeform] = []
    population = make_initial_population(rng, grammar)
    for ind in population:
        nvdb.add_item(ind)

    gen = 0
    while gen != config.GENTRAIN_NUM_GENERATIONS:
        if len(archive) == config.GENTRAIN_ARCHIVE_SIZE:
            logging.info(f"Archive novelty: {[sum(nvdb.novelty_subset(archive))]}")
        else:
            logging.info(
                f"Building archive.. {len(archive)} / {config.GENTRAIN_ARCHIVE_SIZE}"
            )

        maybe_removed: Set[DirectedTreeNodeform] = set()

        # update novelties of existing and new individuals
        to_measure = nvdb.get_unknown_differences()
        measured = measure_distances(to_measure, parallelism)
        nvdb.set_unknown_differences(
            [(items, measure) for items, measure in zip(to_measure, measured)]
        )

        # add most novel individuals from population to archive
        pop_novelties = [nvdb.novelty(ind) for ind in population]
        pop_novelties_sorted = np.argsort(pop_novelties)
        to_archive_indices = pop_novelties_sorted[-config.GENTRAIN_ARCHIVE_APPEND_NUM :]
        archive.extend([population[index] for index in to_archive_indices])

        # # reduce archive to maximum size
        while len(archive) > config.GENTRAIN_ARCHIVE_SIZE:
            archive_novelties = nvdb.novelty_subset(archive)
            remove_index = np.argmin(archive_novelties)
            maybe_removed.add(archive[remove_index])
            archive.pop(remove_index)

        # create offspring and replace in population
        parent_indices = pop_novelties_sorted[-config.GENTRAIN_OFFSPRING_SIZE :]
        offspring = [population[parent_i].copy() for parent_i in parent_indices]
        for individual in offspring:
            individual.mutate(
                rng=rng, grammar=grammar, max_modules=config.MODEL_MAX_MODULES
            )

        replace_indices = [
            random.randrange(len(population)) for _ in range(len(offspring))
        ]
        for child, replace_index in zip(offspring, replace_indices):
            if population[replace_index] not in offspring:
                maybe_removed.add(population[replace_index])
            population[replace_index] = child
        # remove items removed from archive and population if they are not present in both
        for item in maybe_removed:
            if (item not in archive) and (item not in population):
                nvdb.remove_item(item)
        for child, replace_index in zip(offspring, replace_indices):
            if child in population:
                nvdb.add_item(child)

        print(len(population), len(archive), len(nvdb.items), len(nvdb.unknown))

        # go to next generation once archive is full
        if len(archive) == config.GENTRAIN_ARCHIVE_SIZE:
            gen += 1

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
