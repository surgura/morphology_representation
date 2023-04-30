from ...generic.population import Population as GenericPopulation
from .base import Base
from .individual import Individual


class Population(Base, GenericPopulation[Individual]):
    __tablename__ = "population"
