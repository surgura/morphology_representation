from dataclasses import dataclass
from typing import TypeVar, Generic, List, Tuple

TRepresentation = TypeVar("TRepresentation")


@dataclass
class EvaluationSet(Generic[TRepresentation]):
    representations: List[TRepresentation]
    pairs: List[Tuple[TRepresentation, TRepresentation]]
    distances: List[float]
    bins: List[List[Tuple[TRepresentation, TRepresentation, float]]]
    bin_ranges: List[Tuple[float, float]]
