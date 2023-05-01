import argparse
import hashlib
import logging
from typing import List, Tuple

import multineat
import numpy as np
from sqlalchemy.orm import Session

import config
import indices_range
import robot_optimization.benchmark.model as model
from evaluator import Evaluator
from revolve2.core.database import open_database_sqlite
from revolve2.core.optimization.ea.generic_ea import population_management, selection


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
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    parent1: model.Genotype,
    parent2: model.Genotype,
) -> model.Genotype:
    return model.Genotype.crossover(parent1, parent2, rng).mutate(
        innov_db_body, innov_db_brain, rng
    )


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
                original_population.individuals[i].genotype,
                original_population.individuals[i].fitness,
            )
            for i in original_survivors
        ]
        + [
            model.Individual(
                offspring_population.individuals[i].genotype,
                offspring_population.individuals[i].fitness,
            )
            for i in offspring_survivors
        ]
    )


def do_run(run: int, optrun: int, num_simulators: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    rng_seed = int(
        hashlib.sha256(
            f"opt_root_displacement_benchmark_seed{config.OPTBENCH_RNG_SEED}_run{run}_optrun{optrun}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    evaluator = Evaluator(True, num_simulators)

    # multineat innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    dbengine = open_database_sqlite(config.OPTBENCH_OUT(run, optrun), create=True)
    model.Base.metadata.create_all(dbengine)

    logging.info("Generating initial population.")
    initial_genotypes = [
        model.Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
            num_initial_mutations=config.ROBOPT_NUM_INITIAL_MUTATIONS,
        )
        for _ in range(config.ROBOPT_POPULATION_SIZE)
    ]
    logging.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(
        [genotype.develop() for genotype in initial_genotypes]
    )
    population = model.Population(
        [
            model.Individual(genotype, fitness)
            for genotype, fitness in zip(initial_genotypes, initial_fitnesses)
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
        offspring_genotypes = [
            mate(
                rng,
                innov_db_body,
                innov_db_brain,
                generation.population.individuals[parent1_i].genotype,
                generation.population.individuals[parent2_i].genotype,
            )
            for parent1_i, parent2_i in parents
        ]
        offspring_fitnesses = evaluator.evaluate(
            [genotype.develop() for genotype in offspring_genotypes]
        )
        offspring_population = model.Population(
            [
                model.Individual(genotype, fitness)
                for genotype, fitness in zip(offspring_genotypes, offspring_fitnesses)
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
    parser.add_argument(
        "-r",
        "--runs",
        type=indices_range.indices_type(range(config.RUNS)),
        required=True,
    )
    parser.add_argument(
        "--optruns",
        type=indices_range.indices_type(range(config.ROBOPT_RUNS)),
        required=True,
    )
    parser.add_argument("-p", "--parallelism", type=int, default=1)
    args = parser.parse_args()

    for run in args.runs:
        for optrun in args.optruns:
            do_run(run, optrun, args.parallelism)


if __name__ == "__main__":
    main()
