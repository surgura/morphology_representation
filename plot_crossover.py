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


def do_run(experiment_name: str, run: int, t_dim_i: int, r_dim_i: int) -> None:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    grammar = make_body_rgt()
    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(
            config.TRAIN_DD_OUT(
                experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
            )
        )
    )

    for i in range(100):
        torch.manual_seed(i)
        repr1 = torch.rand(r_dim) * 2.0 - 1.0
        repr2 = repr1 + 0.25 * (torch.rand(r_dim) * 2.0 - 1.0).norm()
        tree1 = GraphAdjform(
            *model.decode(repr1, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )
        tree2 = GraphAdjform(
            *model.decode(repr2, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )

        # averaging
        repr3 = torch.mean(torch.stack([repr1, repr2]), dim=0)

        # uniform crossover
        # mask = torch.randint(low=0, high=2, size=repr1.size(), dtype=torch.bool)
        # repr3 = torch.where(mask, repr1, repr2)

        tree3 = GraphAdjform(
            *model.decode(
                repr3,
                max_size=config.MODEL_MAX_MODULES_INCL_EMPTY,
            )[:2]
        )

        out_file1 = config.PLTXOVER_OUT(
            experiment_name=experiment_name,
            run=run,
            t_dim=t_dim,
            r_dim=r_dim,
            tag=f"{i}_a",
        )
        out_file2 = config.PLTXOVER_OUT(
            experiment_name=experiment_name,
            run=run,
            t_dim=t_dim,
            r_dim=r_dim,
            tag=f"{i}_b",
        )
        out_file3 = config.PLTXOVER_OUT(
            experiment_name=experiment_name,
            run=run,
            t_dim=t_dim,
            r_dim=r_dim,
            tag=f"{i}_c",
        )
        pathlib.Path(out_file1).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_file2).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_file3).parent.mkdir(parents=True, exist_ok=True)
        render_modular_robot_radial(tree_to_body(tree1), out_file1)
        render_modular_robot_radial(tree_to_body(tree2), out_file2)
        render_modular_robot_radial(tree_to_body(tree3), out_file3)


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
    args = parser.parse_args()

    for run in args.runs:
        for t_dim_i in args.t_dims:
            for r_dim_i in args.r_dims:
                do_run(
                    experiment_name=args.experiment_name,
                    run=run,
                    t_dim_i=t_dim_i,
                    r_dim_i=r_dim_i,
                )


if __name__ == "__main__":
    main()
