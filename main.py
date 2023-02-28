from robot_rgt import body_to_tree, Tree, make_body_rgt, tree_to_body
from revolve2.standard_resources import modular_robots
from pyvis.network import Network
from rtgae import recursive_tree_grammar_auto_encoder as rtgae_model
import torch
from simulator import simulate


def visualize_tree(tree: Tree) -> None:
    def shape(node: str) -> str:
        if node == "core":
            return "circle"
        elif node == "brick":
            return "box"
        elif node == "active_hinge":
            return "triangleDown"
        elif node == "empty":
            return "star"
        else:
            return "dot"

    net = Network()
    for i, (node, adj) in enumerate(zip(tree[0], tree[1])):
        net.add_node(i, label=node, shape=shape(node))
    for i, (node, adj) in enumerate(zip(tree[0], tree[1])):
        net.add_edges([(i, neigh) for neigh in adj])

    net.show("vis.html")


grammar = make_body_rgt()

training_data = [body_to_tree(body) for body in modular_robots.all()]

model = rtgae_model.TreeGrammarAutoEncoder(grammar, dim=100, dim_vae=8)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)


# import random

# for epoch in range(1000):
#     optimizer.zero_grad()
#     # sample a random tree from the training data
#     i = random.randrange(len(training_data))
#     nodes, adj = training_data[i]
#     # compute the loss on it
#     loss = model.compute_loss(nodes, adj, beta=0.01, sigma_scaling=0.1)
#     print(loss)
#     # compute the gradient
#     loss.backward()
#     # perform an optimizer step
#     optimizer.step()

# torch.save(model.state_dict(), "model.state")
model.load_state_dict(torch.load("model.state"))

h = torch.tensor([0.5] * 8) + 0.05 * torch.randn(8)
nodes, adj, _ = model.decode(h, max_size=32)
body = tree_to_body(Tree(nodes, adj))

simulate(body)
