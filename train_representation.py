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


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(config.TRAIN_RNG_SEED))

    pathlib.Path(config.TRAIN_OUT).parent.mkdir(parents=True, exist_ok=True)

    grammar = make_body_rgt()

    with open(config.FNT_BEST, "rb") as file:
        best_pop: List[DirectedTreeNodeform]
        (best_pop, _, _) = pickle.load(file)

    training_data_not_unique = [tree.to_graph_adjform() for tree in best_pop]
    training_data = []
    for item in training_data_not_unique:
        if item not in training_data:
            training_data.append(item)
    training_data_pqgrams = [tree_to_pqgrams(tree) for tree in training_data]

    model = rtgae_model.TreeGrammarAutoEncoder(
        grammar, dim=config.MODEL_DIM, dim_vae=config.MODEL_DIM_VAE
    )
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

        print(loss)
        # compute the gradient
        loss.backward()
        # perform an optimizer step
        optimizer.step()

    torch.save(model.state_dict(), config.TRAIN_OUT)


if __name__ == "__main__":
    main()
