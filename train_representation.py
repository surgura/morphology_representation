import torch
from rtgae import recursive_tree_grammar_auto_encoder as rtgae_model
import numpy as np
import config
from robot_rgt import make_body_rgt
from typing import List, cast
from tree import DirectedTreeNodeform
import pickle
from pqgrams_util import tree_to_pqgrams
import pathlib
import argparse
import indices_range
import hashlib
import joblib
import logging


def do_run(run: int, t_dim_i: int, r_dim_i: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    rng_seed = int(
        hashlib.sha256(
            f"train_representation_seed{config.TRAIN_RNG_SEED}_run{run}_t_dim{t_dim}_r_dim{r_dim}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    grammar = make_body_rgt()

    with open(config.FNT_BEST(run), "rb") as file:
        best_pop: List[DirectedTreeNodeform]
        (best_pop, _, _) = pickle.load(file)

    training_data_not_unique = [tree.to_graph_adjform() for tree in best_pop]
    training_data = []
    for item in training_data_not_unique:
        if item not in training_data:
            training_data.append(item)
    training_data_pqgrams = [tree_to_pqgrams(tree) for tree in training_data]

    model = rtgae_model.TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    for _ in range(config.TRAIN_EPOCHS):
        optimizer.zero_grad()
        # sample a random tree from the training data
        (anchor_i, other1_i, other2_i) = cast(
            List[int], rng.integers(0, len(training_data), 3)
        )
        anchor = training_data[anchor_i]
        if training_data_pqgrams[anchor_i].edit_distance(
            training_data_pqgrams[other1_i]
        ) > training_data_pqgrams[anchor_i].edit_distance(
            training_data_pqgrams[other2_i]
        ):
            far = training_data[other2_i]
            near = far = training_data[other1_i]
        else:
            far = training_data[other1_i]
            near = far = training_data[other2_i]

        anchor_loss = model.compute_loss(
            anchor.nodes, anchor.adj, beta=0.01, sigma_scaling=0.1
        )
        near_loss = model.compute_loss(
            near.nodes, near.adj, beta=0.01, sigma_scaling=0.1
        )
        far_loss = model.compute_loss(far.nodes, far.adj, beta=0.01, sigma_scaling=0.1)

        triplet_loss = model.compute_triplet_loss(
            anchor.nodes,
            anchor.adj,
            near.nodes,
            near.adj,
            far.nodes,
            far.adj,
            margin=1.0,
        )

        loss = anchor_loss + near_loss + far_loss + triplet_loss

        logging.info(loss)

        # compute the gradient
        loss.backward()
        # perform an optimizer step
        optimizer.step()

    out_dir = config.TRAIN_OUT(run, t_dim, r_dim)
    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=1)
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

    jobs = []
    for run in args.runs:
        for t_dim_i in args.t_dims:
            for r_dim_i in args.r_dims:
                jobs.append(
                    joblib.delayed(do_run)(run=run, t_dim_i=t_dim_i, r_dim_i=r_dim_i)
                )

    joblib.Parallel(n_jobs=args.jobs)(jobs)


if __name__ == "__main__":
    main()
