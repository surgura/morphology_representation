from ...generic.individual import Individual as GenericIndividual
from .base import Base
from .body_genotype import BodyGenotype
from .brain_parameters import BrainParameters


class Individual(
    Base,
    GenericIndividual[BodyGenotype, BrainParameters],
    population_table="population",
):
    __tablename__ = "individual"
