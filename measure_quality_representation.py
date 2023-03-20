import torch
import config
from robot_rgt import make_body_rgt
from rtgae import recursive_tree_grammar_auto_encoder as rtgae_model
from robot_rgt import tree_to_body
from render2d import render_modular_robot2d
from tree import GraphAdjform
import numpy as np
from pqgrams_util import tree_to_pqgrams
import indices_range
import argparse
import hashlib


def measure_one(
    rng: np.random.Generator, model: rtgae_model.TreeGrammarAutoEncoder
) -> float:
    genotype = torch.rand(config.MODEL_DIM_VAE) * 2 - 1
    genotype_near = genotype + torch.normal(
        mean=torch.tensor([0.0] * config.MODEL_DIM_VAE),
        std=torch.tensor([0.1] * config.MODEL_DIM_VAE),
    )
    genotype_far = genotype + torch.normal(
        mean=torch.tensor([0.0] * config.MODEL_DIM_VAE),
        std=torch.tensor([0.5] * config.MODEL_DIM_VAE),
    )

    phenotype = GraphAdjform(*model.decode(genotype, max_size=32)[:2])
    phenotype_near = GraphAdjform(*model.decode(genotype_near, max_size=32)[:2])
    phenotype_far = GraphAdjform(*model.decode(genotype_far, max_size=32)[:2])

    genotype_norm = (genotype + 1) / 2
    genotype_near_norm = (genotype_near + 1) / 2
    genotype_far_norm = (genotype_far + 1) / 2

    near_genotype_dist = torch.cdist(
        genotype_norm.unsqueeze(0), genotype_near_norm.unsqueeze(0), p=2
    )
    far_genotype_dist = torch.cdist(
        genotype_norm.unsqueeze(0), genotype_far_norm.unsqueeze(0), p=2
    )

    near_dist = tree_to_pqgrams(phenotype).edit_distance(
        tree_to_pqgrams(phenotype_near)
    )
    far_dist = tree_to_pqgrams(phenotype).edit_distance(tree_to_pqgrams(phenotype_far))

    return (near_genotype_dist - near_dist) ** 2 + (far_genotype_dist - far_dist) ** 2


def do_run(run: int) -> None:
    rng_seed = int(
        hashlib.sha256(
            f"measure_quality_representation_seed{config.MREP_RNG_SEED}_run{run}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    torch.manual_seed(rng.integers(0, 10e17))

    grammar = make_body_rgt()

    model = rtgae_model.TreeGrammarAutoEncoder(
        grammar, dim=config.MODEL_DIM, dim_vae=config.MODEL_DIM_VAE
    )
    model.load_state_dict(torch.load(config.TRAIN_OUT(run)))

    # score = sum([measure_one(rng, model) for _ in range(config.MREP_NUM_SAMPLES)])
    # print(score)

    l = 30
    x = torch.rand(config.MODEL_DIM_VAE) * 2 - 1
    print(x)
    for i in range(l):
        h = x + torch.normal(
            mean=torch.tensor([0.0] * config.MODEL_DIM_VAE),
            std=torch.tensor([0.1] * config.MODEL_DIM_VAE),
        )
        print(h)
        nodes, adj, _ = model.decode(h, max_size=32)
        body = tree_to_body(GraphAdjform(nodes, adj))
        render_modular_robot2d(body, f"img/{i}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=indices_range.indices_type, required=True)
    args = parser.parse_args()

    for run in args.runs:
        do_run(run)


if __name__ == "__main__":
    main()
