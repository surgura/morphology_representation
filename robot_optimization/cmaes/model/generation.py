import sqlalchemy.orm as orm

from revolve2.core.database import HasId
from .brain_parameters import BrainParameters
from .body_parameters import BodyParameters
from .base import Base
import sqlalchemy
from .sample_pop import SamplePop


class Generation(Base, HasId, orm.MappedAsDataclass):
    __tablename__ = "generation"
    generation_index: orm.Mapped[int] = orm.mapped_column(nullable=False, unique=True)
    performed_evaluations: orm.Mapped[int] = orm.mapped_column(
        nullable=False, unique=True
    )
    body_parameters_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(f"{BodyParameters.__tablename__}.id"),
        nullable=False,
        init=False,
    )
    body_parameters: orm.Mapped[BodyParameters] = orm.relationship()
    fitness_before_learning: orm.Mapped[float] = orm.mapped_column(nullable=False)
    fitness: orm.Mapped[float] = orm.mapped_column(nullable=False)
    brain_parameters_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(f"{BrainParameters.__tablename__}.id"),
        nullable=False,
        init=False,
    )
    brain_parameters: orm.Mapped[BrainParameters] = orm.relationship()
    sample_pop_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(f"{SamplePop.__tablename__}.id"),
        nullable=False,
        init=False,
    )
    sample_pop: orm.Mapped[SamplePop] = orm.relationship()
    cmaes_pickle: orm.Mapped[str] = orm.mapped_column(nullable=False)
