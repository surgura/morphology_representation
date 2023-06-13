from ...generic.individual import Individual as GenericIndividual
from .base import Base
from .pop_body_params import PopBodyParams
from .pop_brain_params import PopBrainParams


class PopIndividual(
    Base,
    GenericIndividual[PopBodyParams, PopBrainParams],
    population_table="sample_pop",
):
    __tablename__ = "pop_individual"
