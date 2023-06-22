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
import pathlib
import math
import hashlib
from torch.nn.functional import normalize
from apted_util import tree_to_apted, apted_tree_edit_distance


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

    rng_seed = (
        int(
            hashlib.sha256(
                f"measure_distance_preservation_seed{config.DPREVRTGAE_RNG_SEED}_rtgae_run{run}_r_dim{r_dim}_margin{margin}_gain{gain}".encode()
            ).hexdigest(),
            16,
        )
        % 2**64
    )
    rng = torch.Generator()
    rng.manual_seed(rng_seed)

    logging.info(
        f"Measuring locality for RTGAE {run=} {t_dim=} {r_dim=} {margin=} {gain=}"
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

    repr_mapped_as_pqgrams = {
        repr: tree_to_apted(
            GraphAdjform(
                *model.decode(repr, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY)[:2]
            )
        )
        for repr in reprset.representations
    }

    dists_in_solspace = []
    dists_in_reprspace = []
    for repr in reprset.representations:
        for _ in range(10):
            neighbor = repr + config.DPREVRTGAE_DIST * torch.rand(
                size=(1,), generator=rng
            ) * normalize(torch.rand(size=(r_dim,), generator=rng) * 2.0 - 1.0, dim=0)

            repr_dist = torch.norm(repr - neighbor).item()
            sol_dist = apted_tree_edit_distance(
                repr_mapped_as_pqgrams[repr],
                tree_to_apted(
                    GraphAdjform(
                        *model.decode(
                            neighbor, max_size=config.MODEL_MAX_MODULES_INCL_EMPTY
                        )[:2]
                    )
                ),
            )

            dists_in_reprspace.append(repr_dist)
            dists_in_solspace.append(sol_dist)

    dist_pairs = list(zip(dists_in_reprspace, dists_in_solspace))

    out_file = config.DPREVRTGAE_OUT(
        experiment_name=experiment_name,
        run=run,
        t_dim=t_dim,
        r_dim=r_dim,
        margin=margin,
        gain=gain,
    )
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(dist_pairs, f)


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
