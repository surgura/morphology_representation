import logging
from typing import List, Tuple
from robot_optimization.benchmark.genotype import Genotype
import config
import multineat
import hashlib
import argparse
import indices_range
import numpy as np
from robot_optimization.evaluator import Evaluator
from revolve2.core.optimization.ea.generic_ea import selection, population_management
from render2d import render_modular_robot2d


def select_parents(
    rng: np.random.Generator,
    population: List[Genotype],
    fitnesses: List[float],
    offspring_size: int,
) -> List[Tuple[int, int]]:
    return [
        selection.multiple_unique(
            2,
            population,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
        )
        for _ in range(offspring_size)
    ]


def mate(
    rng: np.random.Generator,
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    population: List[Genotype],
    parents: List[Tuple[int, int]],
) -> List[Genotype]:
    return [
        Genotype.crossover(population[parent1], population[parent2], rng).mutate(
            innov_db_body, innov_db_brain, rng
        )
        for (parent1, parent2) in parents
    ]


def select_survivors(
    rng: np.random.Generator,
    population: List[Genotype],
    fitnesses: List[float],
    offspring: List[Genotype],
    offspring_fitnesses: List[float],
) -> Tuple[List[Genotype], List[float]]:
    old_survivors, new_survivors = population_management.steady_state(
        population,
        fitnesses,
        offspring,
        offspring_fitnesses,
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return (
        [population[i] for i in old_survivors] + [offspring[i] for i in new_survivors],
        [fitnesses[i] for i in old_survivors]
        + [offspring_fitnesses[i] for i in new_survivors],
    )


def do_run(run: int, num_simulators: int) -> None:
    rng_seed = int(
        hashlib.sha256(
            f"opt_root_displacement_benchmark_seed{config.OPTBENCH_RNG_SEED}_run{run}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    evaluator = Evaluator(True, num_simulators)

    # multineat innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    population = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
            num_initial_mutations=config.ROBOPT_NUM_INITIAL_MUTATIONS,
        )
        for _ in range(config.ROBOPT_POPULATION_SIZE)
    ]
    for i, g in enumerate(population):
        g.i = i
    generation_index = 0
    fitnesses = evaluator.evaluate([genotype.develop() for genotype in population])

    while generation_index < config.ROBOPT_NUM_GENERATIONS:
        print(max(fitnesses))
        print(np.average(fitnesses))

        parents = select_parents(
            rng, population, fitnesses, config.ROBOPT_OFFSPRING_SIZE
        )
        offspring = mate(rng, innov_db_body, innov_db_brain, population, parents)
        offspring_fitnesses = evaluator.evaluate(
            [genotype.develop() for genotype in offspring]
        )
        population, fitnesses = select_survivors(
            rng, population, fitnesses, offspring, offspring_fitnesses
        )

        generation_index += 1


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
    parser.add_argument("-p", "--parallelism", type=int, default=1)
    args = parser.parse_args()

    for run in args.runs:
        do_run(run, args.parallelism)


if __name__ == "__main__":
    main()
