import argparse
import hashlib
import logging
import pathlib
import pickle
from typing import List, Tuple

import joblib
import numpy as np
import torch

import config
import indices_range
from robot_rgt import make_body_rgt
from rtgae import recursive_tree_grammar_auto_encoder as rtgae_model
import matplotlib.pyplot as plt
import pandas
from train_set import TrainSet
from torch.utils.data import DataLoader
from pqgrams import Profile
from tree import DirectedTreeNodeform, GraphAdjform
from pqgrams_util import tree_to_pqgrams


def compute_distance_matrix(graphs: List[GraphAdjform]):
    aspqgrams = [tree_to_pqgrams(tree) for tree in graphs]

    n = len(graphs)

    distance_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = aspqgrams[i].edit_distance(aspqgrams[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def train_epoch(
    model: rtgae_model.TreeGrammarAutoEncoder,
    train_loader: DataLoader[Tuple[DirectedTreeNodeform, GraphAdjform, Profile]],
    optimizer: torch.optim.Optimizer,
):
    train_loss = []
    aggr_recon_loss = []
    aggr_metric_loss = []
    for batch in train_loader:
        graphs = [GraphAdjform(graph["nodes"], graph["adj"]) for graph in batch]

        recon_losses = [
            model.compute_loss(graph.nodes, graph.adj, beta=0.01, sigma_scaling=0.1)
            for graph in graphs
        ]
        recon_loss = sum(recon_losses[1:], recon_losses[0])

        distance_matrix = compute_distance_matrix(graphs)
        sorted = distance_matrix.argsort(dim=1)

        metric_losses = []
        for anchor_i in range(len(graphs)):
            nonzero_argsort_i = 0
            while nonzero_argsort_i < len(graphs):
                if [anchor_i, sorted[anchor_i, nonzero_argsort_i]] != 0:
                    break
                nonzero_argsort_i += 1
            if nonzero_argsort_i == len(graphs):
                logging.info(
                    "Could not find non-zero distanced positive item for anchor."
                )

            positive_argsort_i = torch.randint(
                low=nonzero_argsort_i, high=len(graphs) - 1, size=(1,)
            ).item()
            negative_argsort_i = torch.randint(
                low=positive_argsort_i + 1, high=len(graphs), size=(1,)
            ).item()
            positive_i = sorted[anchor_i, positive_argsort_i]
            negative_i = sorted[anchor_i, negative_argsort_i]
            metric_losses.append(
                model.compute_triplet_loss(
                    anchor_nodes=graphs[anchor_i].nodes,
                    anchor_adj=graphs[anchor_i].adj,
                    positive_nodes=graphs[positive_i].nodes,
                    positive_adj=graphs[positive_i].adj,
                    negative_nodes=graphs[negative_i].nodes,
                    negative_adj=graphs[negative_i].adj,
                    margin=config.TRAIN_DD_MARGIN,
                )
            )
        metric_loss = sum(metric_losses[1:], metric_losses[0])

        loss = recon_loss + config.TRAIN_DD_TRIPLET_FACTOR * metric_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().numpy())

        aggr_recon_loss.append(recon_loss.detach().numpy())
        aggr_metric_loss.append(metric_loss.detach().numpy())
    print(
        f"recon={float(np.mean(aggr_recon_loss))} metric={float(np.mean(aggr_metric_loss))}"
    )
    return float(np.mean(train_loss))


def collate(data):
    _, graph_adj_forms, _ = zip(*data)
    return [
        {"nodes": graph_adj_form.nodes, "adj": graph_adj_form.adj}
        for graph_adj_form in graph_adj_forms
    ]


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

    torch.manual_seed(rng.integers(2e16))

    grammar = make_body_rgt()
    model = rtgae_model.TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.train()

    train_set = TrainSet(run=run, experiment_name=experiment_name)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-05)

    losses = []
    for epoch in range(config.TRAIN_EPOCHS):
        loss = (
            train_epoch(model=model, train_loader=train_loader, optimizer=optimizer)
            / config.TRAIN_BATCH_SIZE
        )
        print(f"{epoch} : {loss=}")
        losses.append(loss)

    out_dir = config.TRAIN_DD_OUT_LOSS(
        experiment_name=experiment_name, run=run, t_dim=t_dim, r_dim=r_dim
    )
    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir, "wb") as f:
        pickle.dump(losses, f)

    out_dir = config.TRAIN_DD_OUT(
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

    joblib.Parallel(n_jobs=args.parallelism)(jobs)


if __name__ == "__main__":
    main()
