import logging
import argparse
import indices_range
import config
import joblib
import hashlib
import numpy as np
import torch
from robot_rgt import make_body_rgt
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from tree import GraphAdjform
from pqgrams_util import tree_to_pqgrams
import pickle
import pathlib


def measure_one(
    rng: np.random.Generator, model: TreeGrammarAutoEncoder, r_dim: int
) -> float:
    genotype = torch.rand(r_dim) * 2 - 1
    genotype_near = genotype + torch.normal(
        mean=torch.tensor([0.0] * r_dim),
        std=torch.tensor([0.1] * r_dim),
    )
    genotype_far = genotype + torch.normal(
        mean=torch.tensor([0.0] * r_dim),
        std=torch.tensor([0.5] * r_dim),
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

    return (near_genotype_dist.item() - near_dist) ** 2 + (
        far_genotype_dist.item() - far_dist
    ) ** 2


def do_run(run: int, t_dim_i: int, r_dim_i: int) -> None:
    t_dim = config.MODEL_T_DIMS[t_dim_i]
    r_dim = config.MODEL_R_DIMS[r_dim_i]

    rng_seed = int(
        hashlib.sha256(
            f"measure_locality_seed{config.MLOC_RNG_SEED}_run{run}_t_dim{t_dim}_r_dim{r_dim}".encode()
        ).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    torch.manual_seed(rng.integers(0, 10e17))

    grammar = make_body_rgt()

    model = TreeGrammarAutoEncoder(grammar, dim=t_dim, dim_vae=r_dim)
    model.load_state_dict(
        torch.load(config.TRAIN_OUT(run=run, t_dim=t_dim, r_dim=r_dim))
    )

    score = sum(
        [measure_one(rng, model, r_dim=r_dim) for _ in range(config.MLOC_NUM_SAMPLES)]
    )

    out_dir = config.MLOC_OUT(run, t_dim, r_dim)
    pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir, "wb") as f:
        pickle.dump(score, f)


# l = 30
# x = torch.rand(config.MODEL_DIM_VAE) * 2 - 1
# for i in range(l):
#     h = x + torch.normal(
#         mean=torch.tensor([0.0] * config.MODEL_DIM_VAE),
#         std=torch.tensor([0.1] * config.MODEL_DIM_VAE),
#     )
#     nodes, adj, _ = model.decode(h, max_size=32)
#     body = tree_to_body(GraphAdjform(nodes, adj))
#     render_modular_robot2d(body, f"img/{i}.png")


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
