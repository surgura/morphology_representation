from pqgrams import Node as PqgramsNode
from pqgrams import Profile as PqgramsProfile
from tree import GraphAdjform


def __tree_to_pqgrams_add_kids(
    tree: GraphAdjform, parent_i: int, parent_node: PqgramsNode
) -> None:
    for kid_i in tree.adj[parent_i]:
        if tree.nodes[kid_i] == "brick":
            kid = PqgramsNode("brick")
            parent_node.addkid(kid)
            __tree_to_pqgrams_add_kids(tree, kid_i, kid)
        elif tree.nodes[kid_i] == "active_hinge":
            kid = PqgramsNode("active_hinge")
            parent_node.addkid(kid)
            __tree_to_pqgrams_add_kids(tree, kid_i, kid)
        elif tree.nodes[kid_i] == "empty":
            kid = PqgramsNode("empty")
            parent_node.addkid(kid)
            __tree_to_pqgrams_add_kids(tree, kid_i, kid)
        elif tree.nodes[kid_i] == "child":
            print("found node 'child'. Interpreting as 'empty'..")
            kid = PqgramsNode("empty")
            parent_node.addkid(kid)
            __tree_to_pqgrams_add_kids(tree, kid_i, kid)
        else:
            raise NotImplementedError()


def tree_to_pqgrams(tree: GraphAdjform) -> PqgramsProfile:
    core = PqgramsNode("core")
    __tree_to_pqgrams_add_kids(tree, 0, core)
    return PqgramsProfile(core)
