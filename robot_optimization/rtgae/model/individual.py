from ...generic.individual import Individual as GenericIndividual
from .base import Base
from .body_representation import BodyRepresentation
from .brain_parameters import BrainParameters


class Individual(
    Base,
    GenericIndividual[BodyRepresentation, BrainParameters],
    population_table="population",
):
    __tablename__ = "individual"
