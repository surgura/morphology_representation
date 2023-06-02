"""Parameter class."""

from .base import Base
from revolve2.core.database import HasId
from ...generic.brain_parameters import BrainParameters as GenericBrainParameters


class BrainParameters(Base, HasId, GenericBrainParameters):
    """SQLAlchemy model for brain parameters."""

    __tablename__ = "brain_parameters"
