from torch.utils.data import Dataset
from tree import DirectedTreeNodeform, GraphAdjform
from typing import List, Tuple
import pickle
import config
from pqgrams import Profile
from pqgrams_util import tree_to_pqgrams
from apted_util import tree_to_apted
import apted
import torch


class TrainSet(Dataset[Tuple[DirectedTreeNodeform, GraphAdjform, Profile]]):
    _tree_node_form: List[DirectedTreeNodeform]
    _graph_adj_form: List[GraphAdjform]
    _pqgrams: List[Profile]
    _apted: List[apted.helpers.Tree]
    distance_matrix: torch.Tensor

    def __init__(self, run: int, experiment_name: str) -> None:
        with open(
            config.GENTRAIN_OUT(run=run, experiment_name=experiment_name), "rb"
        ) as f:
            trainset = pickle.load(f)
            trees = trainset["trees"]
            assert all([isinstance(item, DirectedTreeNodeform) for item in trees])
            self._tree_node_form = trees
            self.distance_matrix = trainset["distance_matrix"]
        self._graph_adj_form = [
            tree.to_graph_adjform() for tree in self._tree_node_form
        ]
        self._pqgrams = [tree_to_pqgrams(tree) for tree in self._graph_adj_form]
        self._apted = [tree_to_apted(tree) for tree in self._graph_adj_form]

    def __len__(self) -> int:
        return len(self._tree_node_form)

    def __getitem__(
        self, index: int
    ) -> Tuple[int, DirectedTreeNodeform, GraphAdjform, Profile, apted.helpers.Tree]:
        return (
            index,
            self._tree_node_form[index],
            self._graph_adj_form[index],
            self._pqgrams[index],
            self._apted[index],
        )
