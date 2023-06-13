from ...generic.population import Population as GenericPopulation
from .base import Base
from .pop_individual import PopIndividual


class SamplePop(Base, GenericPopulation[PopIndividual]):
    __tablename__ = "sample_pop"
