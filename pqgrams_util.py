import pqgrams
from tree import GraphAdjform


def __tree_to_pqgrams_add_kids(
    tree: GraphAdjform, parent_i: int, parent_node: pqgrams.Node
) -> None:
    for kid_i in tree.adj[parent_i]:
        if tree.nodes[kid_i] == "brick":
            kid = pqgrams.Node("brick")
            parent_node.addkid(kid)
            __tree_to_pqgrams_add_kids(tree, kid_i, kid)
        elif tree.nodes[kid_i] == "active_hinge":
            kid = pqgrams.Node("active_hinge")
            parent_node.addkid(kid)
            __tree_to_pqgrams_add_kids(tree, kid_i, kid)
        elif tree.nodes[kid_i] == "empty":
            kid = pqgrams.Node("empty")
            parent_node.addkid(kid)
            __tree_to_pqgrams_add_kids(tree, kid_i, kid)
        else:
            raise NotImplementedError()


def tree_to_pqgrams(tree: GraphAdjform) -> pqgrams.Profile:
    core = pqgrams.Node("core")
    __tree_to_pqgrams_add_kids(tree, 0, core)
    return pqgrams.Profile(core)
