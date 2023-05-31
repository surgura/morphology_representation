from torch.utils.data import Dataset
from tree import DirectedTreeNodeform, GraphAdjform
from typing import List, Tuple
import pickle
import config
from pqgrams import Profile
from pqgrams_util import tree_to_pqgrams


class TrainSet(Dataset[Tuple[DirectedTreeNodeform, GraphAdjform, Profile]]):
    _tree_node_form: List[DirectedTreeNodeform]
    _graph_adj_form: List[GraphAdjform]
    _pqgrams: List[Profile]

    def __init__(self, run: int) -> None:
        with open(config.GENTRAIN_OUT(run), "rb") as f:
            trainset = pickle.load(f)
            assert all([isinstance(item, DirectedTreeNodeform) for item in trainset])
            self._tree_node_form = trainset
        self._graph_adj_form = [
            tree.to_graph_adjform() for tree in self._tree_node_form
        ]
        self._pqgrams = [tree_to_pqgrams(tree) for tree in self._graph_adj_form]

    def __len__(self) -> int:
        return len(self._tree_node_form)

    def __getitem__(
        self, index: int
    ) -> Tuple[DirectedTreeNodeform, GraphAdjform, Profile]:
        return (
            self._tree_node_form[index],
            self._graph_adj_form[index],
            self._pqgrams[index],
        )