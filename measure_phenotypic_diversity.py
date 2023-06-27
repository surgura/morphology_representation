import logging
import argparse
import indices_range
import config
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from robot_rgt import make_body_rgt, body_to_tree
import torch
from evaluation_representation_set import EvaluationRepresentationSet
import pickle
from tree import GraphAdjform
import pathlib
import math
import hashlib
from torch.nn.functional import normalize
from apted_util import tree_to_apted, apted_tree_edit_distance
import robot_optimization.cmaes.model as cmodel
from revolve2.core.database import open_database_sqlite
import robot_optimization.benchmark.model as bmodel
import pandas
from sqlalchemy.orm import Session
from sqlalchemy import select
from opt_robot_displacement_cmaes import representation_to_body
from typing import List
import joblib
import numpy as np
import numpy.typing as npt
import apted.helpers


def measure_distance_matrix(
    generation: List[apted.helpers.Tree],
) -> npt.NDArray[np.float64]:
    n = len(generation)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = apted_tree_edit_distance(
                generation[i], generation[j]
            )
    return distance_matrix


def measure_distance_matrix_multiple(
    generations: List[List[apted.helpers.Tree]],
) -> List[npt.NDArray[np.float64]]:
    return [
        (print(i), measure_distance_matrix(gen))[1] for i, gen in enumerate(generations)
    ]


def measure_distance_matrix_parallel(
    generations: List[List[apted.helpers.Tree]],
    parallelism: int,
) -> List[npt.NDArray[np.float64]]:
    slices = [
        (
            job_i * len(generations) // parallelism,
            (job_i + 1) * len(generations) // parallelism,
        )
        for job_i in range(parallelism)
    ]
    slices[-1] = (slices[-1][0], len(generations))

    results: List[List[float]] = joblib.Parallel(n_jobs=parallelism)(
        [
            joblib.delayed(measure_distance_matrix_multiple)(
                generations[slice[0] : slice[1]]
            )
            for slice in slices
        ]
    )
    return sum(results, [])


def cppn(
    experiment_name: str,
    run: int,
    optrun: int,
    parallelism: int,
) -> None:
    logging.info(f"Measuring phenotypic diversity for CPPN {run=} {optrun=}")

    dbengine = open_database_sqlite(
        config.OPTBENCH_OUT(experiment_name=experiment_name, run=run, optrun=optrun)
    )
    generations: List[List[apted.helpers.Tree]]  # generation -> index -> apted tree
    with Session(dbengine) as ses:
        stmt = (
            select(
                bmodel.Generation.generation_index,
                bmodel.Individual.population_index,
                bmodel.BodyGenotype,
            )
            .join_from(bmodel.Generation, bmodel.Population)
            .join_from(bmodel.Population, bmodel.Individual)
            .join_from(bmodel.Individual, bmodel.BodyGenotype)
            .order_by(
                bmodel.Generation.generation_index,
                bmodel.Individual.population_index,
            )
        )
        rows = ses.execute(stmt)

        generations = [[]]
        for row in rows:
            if row[0] > len(generations):
                generations.append([])
            tree = body_to_tree(row[2].develop())
            generations[-1].append(tree_to_apted(tree))

    distance_matrices = measure_distance_matrix_parallel(generations, parallelism)

    out_file = config.PHENDIV_CPPN_OUT(
        experiment_name=experiment_name,
        run=run,
        optrun=optrun,
    )
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(distance_matrices, f)


def cmaes(
    experiment_name: str,
    run: int,
    optrun: int,
    t_dim: int,
    r_dim: int,
    margin: float,
    gain: float,
    parallelism: int,
) -> None:
    logging.info(
        f"Measuring phenotypic diversity for CMAES {run=} {optrun=} {t_dim=} {r_dim=} {margin=} {gain=}"
    )

    grammar = make_body_rgt()
    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(
            config.TRAIN_DD_OUT(
                experiment_name=experiment_name,
                run=run,
                t_dim=t_dim,
                r_dim=r_dim,
                margin=margin,
                gain=gain,
            )
        )
    )

    dbengine = open_database_sqlite(
        config.OPTCMAES_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
            t_dim=t_dim,
            r_dim=r_dim,
            margin=margin,
            gain=gain,
        )
    )
    generations: List[List[apted.helpers.Tree]]  # generation -> index -> apted tree
    with Session(dbengine) as ses:
        stmt = (
            select(
                cmodel.Generation.generation_index,
                cmodel.PopIndividual.population_index,
                cmodel.PopBodyParams,
            )
            .join_from(cmodel.Generation, cmodel.SamplePop)
            .join_from(cmodel.SamplePop, cmodel.PopIndividual)
            .join_from(cmodel.PopIndividual, cmodel.PopBodyParams)
            .order_by(
                cmodel.Generation.generation_index,
                cmodel.PopIndividual.population_index,
            )
        )
        rows = ses.execute(stmt)

        generations = [[]]
        for row in rows:
            if row[0] > len(generations):
                generations.append([])
            params = row[2].body
            tree = GraphAdjform(
                *model.decode(
                    torch.tensor(params), max_size=config.MODEL_MAX_MODULES_INCL_EMPTY
                )[:2]
            )
            generations[-1].append(tree_to_apted(tree))

    distance_matrices = measure_distance_matrix_parallel(generations, parallelism)

    out_file = config.PHENDIV_CMAES_OUT(
        experiment_name=experiment_name,
        run=run,
        optrun=optrun,
        t_dim=t_dim,
        r_dim=r_dim,
        margin=margin,
        gain=gain,
    )
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(distance_matrices, f)


def vector_sample(
    experiment_name: str,
    run: int,
    t_dim: int,
    r_dim: int,
    margin: float,
    gain: float,
    parallelism: int,
) -> None:
    logging.info(
        f"Measuring phenotypic diversity for random vector samples {run=} {t_dim=} {r_dim=} {margin=} {gain=}"
    )

    grammar = make_body_rgt()
    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(
            config.TRAIN_DD_OUT(
                experiment_name=experiment_name,
                run=run,
                t_dim=t_dim,
                r_dim=r_dim,
                margin=margin,
                gain=gain,
            )
        )
    )

    rng_seed = (
        int(
            hashlib.sha256(
                f"measure_phenotypic_diversity_vectors_seed{config.PHENDIV_SEED}_run{run}_t_dim{t_dim}_r_dim{r_dim}_margin{margin}_gain{gain}".encode()
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
        for _ in range(config.ROBOPT_POPULATION_SIZE)
    ]

    trees = [
        GraphAdjform(
            *model.decode(repr, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )
        for repr in representations
    ]
    apteds = [tree_to_apted(tree) for tree in trees]

    distance_matrices = measure_distance_matrix_parallel([apteds], parallelism)

    out_file = config.PHENDIV_VECTOR_OUT(
        experiment_name=experiment_name,
        run=run,
        t_dim=t_dim,
        r_dim=r_dim,
        margin=margin,
        gain=gain,
    )
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(distance_matrices, f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument("-p", "--parallelism", type=int, required=True)
    parser.add_argument("--repr", type=str, required=True, choices=["cmaes", "cppn"])

    args = parser.parse_args()

    if args.repr == "cmaes":
        for run in range(config.RUNS):
            for t_dim in config.MODEL_T_DIMS:
                for r_dim in config.MODEL_R_DIMS:
                    for margin in config.TRAIN_DD_MARGINS:
                        for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                            vector_sample(
                                experiment_name=args.experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                                margin=margin,
                                gain=gain,
                                parallelism=args.parallelism,
                            )
                            for optrun in range(config.ROBOPT_RUNS):
                                cmaes(
                                    experiment_name=args.experiment_name,
                                    run=run,
                                    optrun=optrun,
                                    t_dim=t_dim,
                                    r_dim=r_dim,
                                    margin=margin,
                                    gain=gain,
                                    parallelism=args.parallelism,
                                )
    elif args.repr == "cppn":
        for run in range(config.RUNS):
            for optrun in range(config.ROBOPT_RUNS):
                cppn(
                    experiment_name=args.experiment_name,
                    run=run,
                    optrun=optrun,
                    parallelism=args.parallelism,
                )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
