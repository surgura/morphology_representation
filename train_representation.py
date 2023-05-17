import argparse
import hashlib
import logging
import pathlib
import pickle
from typing import List, cast

import joblib
import numpy as np
import torch

import config
import indices_range
from pqgrams_util import tree_to_pqgrams
from robot_rgt import make_body_rgt
from rtgae import recursive_tree_grammar_auto_encoder as rtgae_model
from tree import DirectedTreeNodeform
import matplotlib.pyplot as plt


def do_run(experiment_name: str, run: int, t_dim_i: int, r_dim_i: int) -> None:
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

    training_set: List[DirectedTreeNodeform]
    with open(config.GENTRAIN_OUT(run), "rb") as file:
        training_set = pickle.load(file)
    training_set_graph_adjform = [tree.to_graph_adjform() for tree in training_set]
    training_set_pqgrams = [
        tree_to_pqgrams(tree) for tree in training_set_graph_adjform
    ]

    model = rtgae_model.TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    losses: List[float] = []

    for epoch in range(config.TRAIN_EPOCHS):
        logging.info(f"{epoch=}")

        optimizer.zero_grad()
        # sample a random tree from the training data

        all_indices = [
            int(i)
            for i in rng.choice(
                len(training_set), config.TRAIN_BATCH_SIZE, replace=False
            )
        ]
        anchor_index = all_indices[0]
        other_indices = all_indices[1:]
        distances = np.array(
            [
                training_set_pqgrams[anchor_index].edit_distance(
                    training_set_pqgrams[other_index]
                )
                for other_index in other_indices
            ]
        )
        mask_smaller = distances < config.TRAIN_TRIPLET_LABEL_MARGIN
        mask_larger = distances > config.TRAIN_TRIPLET_LABEL_MARGIN
        if (not mask_smaller.any()) or (not mask_larger.any()):
            continue
        positive_index = other_indices[np.argmax(mask_smaller * distances)]
        negative_index = other_indices[np.argmin(mask_larger * distances)]

        anchor = training_set_graph_adjform[anchor_index]
        positive = training_set_graph_adjform[positive_index]
        negative = training_set_graph_adjform[negative_index]

        anchor_loss = model.compute_loss(
            anchor.nodes,
            anchor.adj,
            beta=0.01,
            sigma_scaling=0.1,
        )
        positive_loss = model.compute_loss(
            positive.nodes,
            positive.adj,
            beta=0.01,
            sigma_scaling=0.1,
        )
        negative_loss = model.compute_loss(
            negative.nodes,
            negative.adj,
            beta=0.01,
            sigma_scaling=0.1,
        )

        triplet_loss = model.compute_triplet_loss(
            anchor.nodes,
            anchor.adj,
            negative.nodes,
            negative.adj,
            positive.nodes,
            positive.adj,
            margin=0.05,
        )

        loss = (
            anchor_loss
            + positive_loss
            + negative_loss
            + config.TRAIN_TRIPLET_FACTOR * triplet_loss
        )

        # compute the gradient
        loss.backward()
        losses.append(float(loss))
        # perform an optimizer step
        optimizer.step()

    plt.plot([i for i in range(len(losses))], losses)
    out_dir = config.TRAIN_OUT_PLOT(
        experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
    )
    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir)

    out_dir = config.TRAIN_OUT(
        experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
    )
    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
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
                    joblib.delayed(do_run)(
                        experiment_name=args.experiment_name,
                        run=run,
                        t_dim_i=t_dim_i,
                        r_dim_i=r_dim_i,
                    )
                )

    joblib.Parallel(n_jobs=args.jobs)(jobs)


if __name__ == "__main__":
    main()
