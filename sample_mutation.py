import argparse
import logging
import pathlib

import config
import indices_range
from render2d import render_modular_robot_radial
from robot_rgt import tree_to_body
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from robot_rgt import make_body_rgt
import torch
from tree import GraphAdjform
from torch.nn.functional import normalize
from evaluator import Evaluator
from robot_rgt import tree_to_body
from make_brain import make_brain
from robot_to_actor_cpg import robot_to_actor_cpg
from revolve2.core.modular_robot import ModularRobot


def do_run(
    experiment_name: str,
    run: int,
    t_dim_i: int,
    r_dim_i: int,
    margin_i: int,
    gain_i: int,
) -> None:
    evaluator = Evaluator(headless=False, num_simulators=1)

    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]
    margin = config.TRAIN_DD_MARGINS[margin_i]
    gain = config.TRAIN_DD_TRIPLET_FACTORS[gain_i]

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

    for i in range(100):
        torch.manual_seed(i)
        repr1 = torch.rand(r_dim) * 2.0 - 1.0
        repr2 = repr1 + 0.1 * (torch.rand(size=(r_dim,)) * 2.0 - 1.0)
        # This is what I want but my supervisor does not understand so I will just do what he says
        # repr2 = repr1 + 0.25 * normalize(torch.rand(size=(r_dim,)) * 2.0 - 1.0, dim=0)
        tree1 = GraphAdjform(
            *model.decode(repr1, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )
        tree2 = GraphAdjform(
            *model.decode(repr2, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )

        out_file1 = config.SMPLMUT_OUT(
            experiment_name=experiment_name,
            run=run,
            t_dim=t_dim,
            r_dim=r_dim,
            margin=margin,
            gain=gain,
            tag=f"{i}_a",
        )
        out_file2 = config.SMPLMUT_OUT(
            experiment_name=experiment_name,
            run=run,
            t_dim=t_dim,
            r_dim=r_dim,
            margin=margin,
            gain=gain,
            tag=f"{i}_b",
        )
        pathlib.Path(out_file1).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_file2).parent.mkdir(parents=True, exist_ok=True)
        render_modular_robot_radial(tree_to_body(tree1), out_file1)
        render_modular_robot_radial(tree_to_body(tree2), out_file2)

        body1 = tree_to_body(tree1)
        body2 = tree_to_body(tree2)
        cpgstuff1 = robot_to_actor_cpg(body1)[1]
        cpgstuff2 = robot_to_actor_cpg(body2)[1]
        brain1 = make_brain(
            cpgstuff1,
            [0.5] * cpgstuff1.num_connections,
        )
        brain2 = make_brain(
            cpgstuff1,
            [0.5] * cpgstuff1.num_connections,
        )
        bot1 = ModularRobot(body1, brain1)
        bot2 = ModularRobot(body2, brain2)

        evaluator.evaluate([bot1])
        evaluator.evaluate([bot2])


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
        "--margins",
        type=indices_range.indices_type(range(len(config.TRAIN_DD_MARGINS))),
        required=True,
    )
    parser.add_argument(
        "--gains",
        type=indices_range.indices_type(range(len(config.TRAIN_DD_TRIPLET_FACTORS))),
        required=True,
    )
    args = parser.parse_args()

    for run in args.runs:
        for t_dim_i in args.t_dims:
            for r_dim_i in args.r_dims:
                for margin_i in args.margins:
                    for gain_i in args.gains:
                        do_run(
                            experiment_name=args.experiment_name,
                            run=run,
                            t_dim_i=t_dim_i,
                            r_dim_i=r_dim_i,
                            margin_i=margin_i,
                            gain_i=gain_i,
                        )


if __name__ == "__main__":
    main()
