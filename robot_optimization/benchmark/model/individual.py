from ...generic.individual import Individual as GenericIndividual
from .base import Base
from .genotype import Genotype


class Individual(Base, GenericIndividual[Genotype], population_table="population"):
    __tablename__ = "individual"
