import argparse
import logging

import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
import robot_optimization.benchmark.model as bmodel
import robot_optimization.rtgae.model as rmodel
import robot_optimization.cmaes.model as cmodel
from evaluator import Evaluator
from revolve2.core.database import open_database_sqlite
from revolve2.core.modular_robot import ModularRobot
from robot_rgt import make_body_rgt
from rtgae import tree_grammar
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from make_brain import make_brain
from robot_to_actor_cpg import robot_to_actor_cpg
from opt_robot_displacement_cmaes import representation_to_body
import pathlib
from render2d import render_modular_robot_radial


def load_robot_cppn(experiment_name: str, run: int, optrun: int) -> ModularRobot:
    dbengine = open_database_sqlite(
        config.OPTBENCH_OUT(experiment_name=experiment_name, run=run, optrun=optrun)
    )
    gen = 10
    with Session(dbengine) as ses:
        first_gen = ses.scalars(
            select(bmodel.Individual)
            .join(bmodel.Population)
            .join(bmodel.Generation)
            .where(bmodel.Generation.generation_index == gen)
            .order_by(
                bmodel.Generation.generation_index.desc(),
            )
        )

        bodies = [row.genotype.develop() for row in first_gen]

    for i, body in enumerate(bodies):
        out_file = f"./teststuff/gen_{gen}/cppn/{i}.png"
        pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        render_modular_robot_radial(body, out_file)


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


def load_robot_rtgae(
    experiment_name: str,
    run: int,
    optrun: int,
    t_dim_i: int,
    r_dim_i: int,
) -> ModularRobot:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    gen = 20
    grammar = make_body_rgt()
    body_model = load_body_model(
        experiment_name=experiment_name,
        run=run,
        t_dim=t_dim,
        r_dim=r_dim,
        grammar=grammar,
    )

    dbengine = open_database_sqlite(
        config.OPTRTGAE_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
            t_dim=t_dim,
            r_dim=r_dim,
        )
    )
    with Session(dbengine) as ses:
        first_gen = ses.scalars(
            select(rmodel.Individual)
            .join(rmodel.Population)
            .join(rmodel.Generation)
            .where(rmodel.Generation.generation_index == gen)
            .order_by(
                rmodel.Generation.generation_index.desc(),
            )
        )

        bodies = [row.genotype.develop(body_model) for row in first_gen]

    for i, body in enumerate(bodies):
        out_file = f"./teststuff/gen_{gen}/rtgae/{i}.png"
        pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        render_modular_robot_radial(body, out_file)


def load_robot_cmaes(
    experiment_name: str,
    run: int,
    optrun: int,
    t_dim_i: int,
    r_dim_i: int,
) -> ModularRobot:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    grammar = make_body_rgt()
    body_model = load_body_model(
        experiment_name=experiment_name,
        run=run,
        t_dim=t_dim,
        r_dim=r_dim,
        grammar=grammar,
    )

    dbengine = open_database_sqlite(
        config.OPTCMAES_OUT(
            experiment_name=experiment_name,
            run=run,
            optrun=optrun,
            t_dim=t_dim,
            r_dim=r_dim,
        )
    )
    with Session(dbengine) as ses:
        last_gen = ses.scalar(
            select(cmodel.Generation)
            .order_by(
                rmodel.Generation.generation_index.desc(),
            )
            .limit(1)
        )
        assert last_gen is not None

        print(f"Fitness from database: {last_gen.fitness}")
        body = representation_to_body(last_gen.body_parameters.parameters, body_model)
        brain = make_brain(
            robot_to_actor_cpg(body)[1],
            last_gen.brain_parameters.parameters,
        )
        return ModularRobot(body, brain)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument(
        "-r", "--run", type=int, choices=range(config.RUNS), required=True
    )
    parser.add_argument(
        "--optrun", type=int, choices=range(config.ROBOPT_RUNS), required=True
    )
    subparsers = parser.add_subparsers(dest="opt", required=True)
    subparsers.add_parser("cppn")
    rtgae_parser = subparsers.add_parser("rtgae")
    rtgae_parser.add_argument(
        "--r_dim",
        type=int,
        choices=range(len(config.MODEL_R_DIMS)),
        required=True,
    )
    rtgae_parser.add_argument(
        "--t_dim",
        type=int,
        choices=range(len(config.MODEL_T_DIMS)),
        required=True,
    )
    cmaes_parser = subparsers.add_parser("cmaes")
    cmaes_parser.add_argument(
        "--r_dim",
        type=int,
        choices=range(len(config.MODEL_R_DIMS)),
        required=True,
    )
    cmaes_parser.add_argument(
        "--t_dim",
        type=int,
        choices=range(len(config.MODEL_T_DIMS)),
        required=True,
    )
    args = parser.parse_args()

    if args.opt == "cppn":
        robot = load_robot_cppn(
            experiment_name=args.experiment_name, run=args.run, optrun=args.optrun
        )
    elif args.opt == "rtgae":
        robot = load_robot_rtgae(
            experiment_name=args.experiment_name,
            run=args.run,
            optrun=args.optrun,
            r_dim_i=args.r_dim,
            t_dim_i=args.t_dim,
        )
    elif args.opt == "cmaes":
        robot = load_robot_cmaes(
            experiment_name=args.experiment_name,
            run=args.run,
            optrun=args.optrun,
            r_dim_i=args.r_dim,
            t_dim_i=args.t_dim,
        )
    else:
        raise NotImplementedError()

    # evaluator = Evaluator(headless=False, num_simulators=1)
    # evaluator.evaluate([robot])


if __name__ == "__main__":
    main()
