"""Parameter class."""

from .base import Base
from revolve2.core.database import HasId
from ...generic.brain_parameters import BrainParameters as GenericBrainParameters


class BodyParameters(Base, HasId, GenericBrainParameters):
    """SQLAlchemy model for body parameters."""

    __tablename__ = "body_parameters"
