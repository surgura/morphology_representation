import sqlalchemy.orm as orm
import sqlalchemy
from .base import Base
from .population import Population


class Generation(Base):
    __tablename__ = "generation"

    id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    generation_index: orm.Mapped[int] = orm.mapped_column(nullable=False, unique=True)
    population_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("population.id"), nullable=False, init=False
    )
    population: orm.Mapped[Population] = orm.relationship()
