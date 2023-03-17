import torch
import config
from robot_rgt import make_body_rgt
from rtgae import recursive_tree_grammar_auto_encoder as rtgae_model
from robot_rgt import tree_to_body
from render2d import render_modular_robot2d
from tree import GraphAdjform


def main() -> None:
    grammar = make_body_rgt()

    model = rtgae_model.TreeGrammarAutoEncoder(
        grammar, dim=config.MODEL_DIM, dim_vae=config.MODEL_DIM_VAE
    )
    model.load_state_dict(torch.load(config.TRAIN_OUT))

    l = 30
    x = torch.rand(config.MODEL_DIM_VAE)
    print(x)
    for i in range(l):
        h = x + torch.normal(
            mean=torch.tensor([0.0] * config.MODEL_DIM_VAE),
            std=torch.tensor([0.1] * config.MODEL_DIM_VAE),
        )
        print(h)
        nodes, adj, _ = model.decode(h, max_size=32)
        body = tree_to_body(GraphAdjform(nodes, adj))
        render_modular_robot2d(body, f"img2/{i}.png")


if __name__ == "__main__":
    main()
