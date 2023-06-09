import argparse
import hashlib
import logging
from typing import List, Tuple

import numpy as np
import torch
from sqlalchemy.orm import Session

import config
import indices_range
import robot_optimization.rtgae.model as model
from evaluator import Evaluator
from revolve2.core.database import open_database_sqlite
from revolve2.core.optimization.ea.generic_ea import population_management, selection
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
import brain_optimizer
from revolve2.core.modular_robot import ModularRobot
from robot_to_actor_cpg import robot_to_actor_cpg
from make_brain import make_brain


def select_parents(
    rng: np.random.Generator,
    population: model.Population,
    offspring_size: int,
) -> List[Tuple[int, int]]:
    return [
        selection.multiple_unique(
            2,
            [individual.genotype for individual in population.individuals],
            [individual.fitness for individual in population.individuals],
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
        )
        for _ in range(offspring_size)
    ]


def mate(
    rng: np.random.Generator,
    parent1: model.BodyRepresentation,
    parent2: model.BodyRepresentation,
) -> model.BodyRepresentation:
    return model.BodyRepresentation.crossover(parent1, parent2, rng).mutate(rng)


def select_survivors(
    rng: np.random.Generator,
    original_population: model.Population,
    offspring_population: model.Population,
) -> model.Population:
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in original_population.individuals],
        [i.fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.fitness for i in offspring_population.individuals],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return model.Population(
        [
            model.Individual(
                genotype=original_population.individuals[i].genotype,
                fitness_before_learning=original_population.individuals[
                    i
                ].fitness_before_learning,
                fitness=original_population.individuals[i].fitness,
                brain_parameters=original_population.individuals[i].brain_parameters,
            )
            for i in original_survivors
        ]
        + [
            model.Individual(
                genotype=offspring_population.individuals[i].genotype,
                fitness_before_learning=offspring_population.individuals[
                    i
                ].fitness_before_learning,
                fitness=offspring_population.individuals[i].fitness,
                brain_parameters=offspring_population.individuals[i].brain_parameters,
            )
            for i in offspring_survivors
        ]
    )


def load_body_model(
    experiment_name: str,
    run: int,
    t_dim: int,
    r_dim: int,
    grammar: tree_grammar.TreeGrammar,
) -> TreeGrammarAutoEncoder:
    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(
            config.TRAIN_OUT(
                experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
            )
        )
    )
    return model


def do_run(
    experiment_name: str,
    run: int,
    t_dim_i: int,
    r_dim_i: int,
    optrun: int,
    parallelism: int,
) -> None:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    rng_seed = int(
        hashlib.sha256(
            f"opt_root_displacement_benchmark_seed{config.OPTRTGAE_RNG_SEED}_run{run}_t_dim{t_dim}_r_dim{r_dim}_optrun{optrun}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    logging.info(f"Running run{run} t_dim{t_dim} r_dim{r_dim} optrun{optrun}")

    grammar = make_body_rgt()
    body_model = load_body_model(
        experiment_name=experiment_name,
        run=run,
        t_dim=t_dim,
        r_dim=r_dim,
        grammar=grammar,
    )

    evaluator = Evaluator(True, parallelism)

    dbengine = open_database_sqlite(
        config.OPTRTGAE_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
            t_dim=t_dim,
            r_dim=r_dim,
        ),
        create=True,
    )
    model.Base.metadata.create_all(dbengine)

    logging.info("Generating initial population.")
    initial_genotypes = [
        model.BodyRepresentation.random(
            rng=rng,
            representation_size=r_dim,
        )
        for _ in range(config.ROBOPT_POPULATION_SIZE)
    ]
    logging.info("Evaluating initial population.")
    initial_bodies = [genotype.develop(body_model) for genotype in initial_genotypes]
    (
        initial_fitnesses_before_learning,
        initial_fitnesses_after_learning,
        initial_optimized_brain_parameters,
    ) = zip(
        *[
            (fitness_before, fitness_after, model.BrainParameters(params))
            for fitness_before, fitness_after, params in brain_optimizer.optimize_multiple_parallel(
                evaluator, rng, initial_bodies, parallelism=(parallelism // 5)
            )
        ]
    )
    initial_optimized_brain_parameters = [
        model.BrainParameters(b)
        for b in brain_optimizer.optimize_multiple_parallel(
            evaluator, rng, initial_bodies, parallelism=(parallelism // 5)
        )
    ]
    population = model.Population(
        [
            model.Individual(
                genotype=genotype,
                fitness_before_learning=fitness_before_learning,
                fitness=fitness_after_learning,
                brain_parameters=brain_parameters,
            )
            for genotype, fitness_before_learning, fitness_after_learning, brain_parameters in zip(
                initial_genotypes,
                initial_fitnesses_before_learning,
                initial_fitnesses_after_learning,
                initial_optimized_brain_parameters,
            )
        ]
    )
    generation = model.Generation(
        0,
        population,
    )
    logging.info("Saving initial population.")
    with Session(dbengine, expire_on_commit=False) as ses:
        ses.add(generation)
        ses.commit()

    logging.info("Start optimization process.")
    while generation.generation_index < config.ROBOPT_NUM_GENERATIONS:
        logging.info(
            f"Generation {generation.generation_index + 1} / {config.ROBOPT_NUM_GENERATIONS}."
        )
        parents = select_parents(
            rng, generation.population, config.ROBOPT_OFFSPRING_SIZE
        )
        offspring_genotypes: List[model.BodyRepresentation] = [
            mate(
                rng,
                generation.population.individuals[parent1_i].genotype,
                generation.population.individuals[parent2_i].genotype,
            )
            for parent1_i, parent2_i in parents
        ]

        offspring_bodies = [
            genotype.develop(body_model) for genotype in offspring_genotypes
        ]
        (
            offspring_fitnesses_before_learning,
            offspring_fitnesses_after_learning,
            offspring_optimized_brain_parameters,
        ) = zip(
            *[
                (fitness_before, fitness_after, model.BrainParameters(params))
                for fitness_before, fitness_after, params in brain_optimizer.optimize_multiple_parallel(
                    evaluator, rng, offspring_bodies, parallelism=(parallelism // 5)
                )
            ]
        )
        offspring_population = model.Population(
            [
                model.Individual(
                    genotype=genotype,
                    fitness_before_learning=fitness_before_learning,
                    fitness=fitness_after_learning,
                    brain_parameters=brain_parameters,
                )
                for genotype, fitness_before_learning, fitness_after_learning, brain_parameters in zip(
                    offspring_genotypes,
                    offspring_fitnesses_before_learning,
                    offspring_fitnesses_after_learning,
                    offspring_optimized_brain_parameters,
                )
            ]
        )
        survived_population = select_survivors(
            rng,
            generation.population,
            offspring_population,
        )
        generation = model.Generation(
            generation.generation_index + 1,
            survived_population,
        )
        with Session(dbengine, expire_on_commit=False) as ses:
            ses.add(generation)
            ses.commit()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
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
    parser.add_argument(
        "--t_dims",
        type=indices_range.indices_type(range(len(config.MODEL_T_DIMS))),
        required=True,
    )
    parser.add_argument(
        "--optruns",
        type=indices_range.indices_type(range(config.ROBOPT_RUNS)),
        required=True,
    )
    parser.add_argument("-p", "--parallelism", type=int, required=True)
    args = parser.parse_args()

    for run in args.runs:
        for optrun in args.optruns:
            for t_dim_i in args.t_dims:
                for r_dim_i in args.r_dims:
                    do_run(
                        experiment_name=args.experiment_name,
                        run=run,
                        t_dim_i=t_dim_i,
                        r_dim_i=r_dim_i,
                        optrun=optrun,
                        parallelism=args.parallelism,
                    )


if __name__ == "__main__":
    main()
