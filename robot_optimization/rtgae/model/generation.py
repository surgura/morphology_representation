from ...generic.generation import Generation as GenericGeneration
from .base import Base
from .population import Population


class Generation(Base, GenericGeneration[Population]):
    __tablename__ = "generation"
