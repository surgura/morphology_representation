import logging
import argparse
import indices_range
import config
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from robot_rgt import make_body_rgt
import torch
from evaluation_representation_set import EvaluationRepresentationSet
import pickle
from tree import GraphAdjform
from apted_util import tree_to_apted, apted_tree_edit_distance
from tree import DirectedTreeNodeform
from typing import List, Tuple
import numpy as np
import joblib
import pathlib
import apted.helpers


def smallest_distance_nonzero(
    tree: apted.helpers.Tree, compare_to: List[apted.helpers.Tree]
) -> float:
    return min(
        [
            x
            for x in [apted_tree_edit_distance(tree, other) for other in compare_to]
            if not x is np.isclose(x, 0.0)
        ]
    )


def smallest_distance_nonzero_multiple(
    trees: List[apted.helpers.Tree],
    compare_to: List[apted.helpers.Tree],
    slice: Tuple[int, int],
) -> List[float]:
    return [
        (
            print(i),
            smallest_distance_nonzero(tree, compare_to),
        )[1]
        for i, tree in enumerate(trees[slice[0] : slice[1]])
    ]


def do_run(
    experiment_name: str,
    run: int,
    t_dim_i: int,
    r_dim_i: int,
    parallelism: int,
    margin_i: int,
    gain_i: int,
) -> None:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]
    margin = config.TRAIN_DD_MARGINS[margin_i]
    gain = config.TRAIN_DD_TRIPLET_FACTORS[gain_i]

    logging.info(
        f"Measuring coverage for RTGAE {run=} {t_dim=} {r_dim=} {margin=} {gain=}"
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

    reprset: EvaluationRepresentationSet[torch.Tensor]
    with open(
        config.GENEVALREPR_OUT_RTGAE(
            run=run, experiment_name=experiment_name, r_dim=r_dim
        ),
        "rb",
    ) as f:
        reprset = pickle.load(f)

    mappeds = [
        GraphAdjform(
            *model.decode(repr, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
        )
        for repr in reprset.representations
    ]
    mapped_as_pqgrams = [tree_to_apted(mapped) for mapped in mappeds]

    solset: List[DirectedTreeNodeform]
    with open(
        config.GENEVALSOL_OUT(run=run, experiment_name=experiment_name), "rb"
    ) as f:
        solset = pickle.load(f)
    sol_as_pqgrams = [tree_to_apted(sol.to_graph_adjform()) for sol in solset]

    slices = [
        (
            job_i * len(sol_as_pqgrams) // parallelism,
            (job_i + 1) * len(sol_as_pqgrams) // parallelism,
        )
        for job_i in range(parallelism)
    ]
    slices[-1] = (slices[-1][0], len(sol_as_pqgrams))
    results: List[List[float]] = joblib.Parallel(n_jobs=parallelism)(
        [
            joblib.delayed(smallest_distance_nonzero_multiple)(
                sol_as_pqgrams, mapped_as_pqgrams, slice
            )
            for slice in slices
        ]
    )
    results_combined = sum(results, [])
    coverage = sum([r**2 for r in results_combined], 0.0)

    out_file = config.CVGRTGAE_OUT(
        experiment_name=experiment_name,
        run=run,
        t_dim=t_dim,
        r_dim=r_dim,
        margin=margin,
        gain=gain,
    )
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(coverage, f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument("-p", "--parallelism", type=int, required=True)
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
                            parallelism=args.parallelism,
                            margin_i=margin_i,
                            gain_i=gain_i,
                        )


if __name__ == "__main__":
    main()
