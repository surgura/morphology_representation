import argparse
import hashlib
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from sqlalchemy.orm import Session

import config
import indices_range
import robot_optimization.cmaes.model as model
from evaluator import Evaluator
from revolve2.core.database import open_database_sqlite
from revolve2.core.optimization.ea.generic_ea import population_management, selection
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
import brain_optimizer
from revolve2.core.modular_robot import ModularRobot, Body
from robot_to_actor_cpg import robot_to_actor_cpg
from make_brain import make_brain
import cma
from robot_rgt import tree_to_body
from tree import GraphAdjform


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


def representation_to_body(representation: Tuple[float, ...], body_model) -> Body:
    nodes, adj, _ = body_model.decode(
        torch.tensor(representation), max_size=config.MODEL_MAX_MODULES_INCL_EMPTY
    )
    return tree_to_body(GraphAdjform(nodes, adj))


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
            f"opt_root_displacement_benchmark_seed{config.OPTCMAES_RNG_SEED}_run{run}_t_dim{t_dim}_r_dim{r_dim}_optrun{optrun}".encode()
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
        config.OPTCMAES_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
            t_dim=t_dim,
            r_dim=r_dim,
        ),
        create=True,
    )
    model.Base.metadata.create_all(dbengine)

    initial_body = r_dim * [0.0]

    options = cma.CMAOptions()
    options.set("seed", rng.integers(0, 2**15))
    options.set("bounds", [-1.0, 1.0])
    opt = cma.CMAEvolutionStrategy(
        initial_body, config.OPTCMAES_BRAIN_INITIAL_STD, options
    )

    last_best: Optional[
        Tuple[Tuple[float, ...], Tuple[float, ...]]
    ] = None  # best body, brain params

    performed_evals = 0
    gen = 0
    while performed_evals < config.OPTCMAES_NUM_EVALUATIONS:
        logging.info(
            f"Evals done {performed_evals} / {config.OPTCMAES_NUM_EVALUATIONS}."
        )

        solutions = [tuple(float(p) for p in params) for params in opt.ask()]
        bodies = [
            representation_to_body(solution, body_model) for solution in solutions
        ]
        optimized_brain_parameters = [
            model.BrainParameters(b)
            for b in brain_optimizer.optimize_multiple_parallel(
                evaluator, rng, bodies, parallelism=(parallelism // 5)
            )
        ]
        modular_robots = [
            ModularRobot(
                body, make_brain(robot_to_actor_cpg(body)[1], brain_genotype.parameters)
            )
            for body, brain_genotype in zip(bodies, optimized_brain_parameters)
        ]
        fitnesses = [
            -1.0 * x for x in evaluator.evaluate([robot for robot in modular_robots])
        ]
        opt.tell(solutions, fitnesses)

        performed_evals += len(solutions)
        gen += 1

        currentbestx = tuple(float(x) for x in opt.result.xbest)
        if last_best is None or last_best[0] != currentbestx:
            idx = solutions.index(currentbestx)
            currentbestbrain = optimized_brain_parameters[idx].parameters
            last_best = (currentbestx, currentbestbrain)

        generation = model.Generation(
            generation_index=gen,
            performed_evaluations=performed_evals,
            body_parameters=model.BodyParameters(last_best[0]),
            fitness=-opt.result.fbest,
            brain_parameters=model.BrainParameters(last_best[1]),
        )
        with Session(dbengine, expire_on_commit=False) as ses:
            ses.add(generation)
            ses.commit()

    logging.info("Body optimization done.")


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
    parser.add_argument("-p", "--parallelism", type=int, default=1)
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
