from dataclasses import dataclass
from typing import Generic, List, Tuple, TypeVar

TRepresentation = TypeVar("TRepresentation")


@dataclass
class EvaluationRepresentationSet(Generic[TRepresentation]):
    representations: List[TRepresentation]
    pairs: List[Tuple[TRepresentation, TRepresentation]]
    distances: List[float]
    bins: List[List[Tuple[TRepresentation, TRepresentation, float]]]
    bin_ranges: List[Tuple[float, float]]
