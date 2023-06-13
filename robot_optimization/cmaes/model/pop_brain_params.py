"""Parameter class."""

from .base import Base
from revolve2.core.database import HasId
from ...generic.brain_parameters import BrainParameters as GenericBrainParameters


class PopBrainParams(Base, HasId, GenericBrainParameters):
    """SQLAlchemy model for parameters."""

    __tablename__ = "pop_brain_params"
