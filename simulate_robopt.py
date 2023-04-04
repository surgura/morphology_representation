import logging
import argparse
import config
from revolve2.core.modular_robot import ModularRobot
from sqlalchemy import select
import robot_optimization.benchmark.model as bmodel
import robot_optimization.rtgae.model as rmodel
from revolve2.core.database import open_database_sqlite
from sqlalchemy.orm import Session
from evaluator import Evaluator
from select_representations import Measure
import pickle
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from rtgae import tree_grammar
import torch
from robot_rgt import make_body_rgt


def load_robot_bench(run: int, optrun: int) -> ModularRobot:
    dbengine = open_database_sqlite(config.OPTBENCH_OUT(run, optrun))
    with Session(dbengine) as ses:
        best_individual_last_gen = ses.scalar(
            select(bmodel.Individual)
            .join(bmodel.Population)
            .join(bmodel.Generation)
            .order_by(
                bmodel.Generation.generation_index.desc(),
                bmodel.Individual.fitness.desc(),
            )
            .limit(1)
        )
        assert best_individual_last_gen is not None

        print(f"Fitness from database: {best_individual_last_gen.fitness}")
        return best_individual_last_gen.genotype.develop()


def load_body_model(
    run: int, t_dim: int, r_dim: int, grammar: tree_grammar.TreeGrammar
) -> TreeGrammarAutoEncoder:
    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(config.TRAIN_OUT(run=run, t_dim=t_dim, r_dim=r_dim))
    )
    return model


def load_robot_rtgae(run: int, optrun: int, bestorworst: bool) -> ModularRobot:
    with open(config.SREP_OUT(run), "rb") as f:
        selection: Measure = pickle.load(f)["best" if bestorworst else "worst"]
    t_dim = selection.t_dim
    r_dim = selection.r_dim

    grammar = make_body_rgt()
    body_model = load_body_model(run=run, t_dim=t_dim, r_dim=r_dim, grammar=grammar)

    dbengine = open_database_sqlite(config.OPTRTGAE_OUT(run, optrun, bestorworst))
    with Session(dbengine) as ses:
        best_individual_last_gen = ses.scalar(
            select(rmodel.Individual)
            .join(rmodel.Population)
            .join(rmodel.Generation)
            .order_by(
                rmodel.Generation.generation_index.desc(),
                rmodel.Individual.fitness.desc(),
            )
            .limit(1)
        )
        assert best_individual_last_gen is not None

        print(f"Fitness from database: {best_individual_last_gen.fitness}")
        return best_individual_last_gen.genotype.develop(body_model)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run", type=int, choices=range(config.RUNS), required=True
    )
    parser.add_argument(
        "--optrun", type=int, choices=range(config.ROBOPT_RUNS), required=True
    )
    subparsers = parser.add_subparsers(dest="opt", required=True)
    subparsers.add_parser("bench")
    rtgae_parser = subparsers.add_parser("rtgae")
    rtgae_parser.add_argument("bestorworst", type=str, choices=["best", "worst"])
    args = parser.parse_args()

    if args.opt == "bench":
        robot = load_robot_bench(args.run, args.optrun)
    else:
        robot = load_robot_rtgae(args.run, args.optrun, args.bestorworst == "best")

    evaluator = Evaluator(headless=False, num_simulators=1)
    evaluator.evaluate([robot])


if __name__ == "__main__":
    main()
