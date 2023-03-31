import sqlalchemy.orm as orm
import sqlalchemy
from .base import Base
from .genotype import Genotype


class Individual(Base):
    __tablename__ = "individual"

    id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    population_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("population.id"), nullable=False, init=False
    )
    population_index: orm.Mapped[int] = orm.mapped_column(nullable=False, init=False)
    genotype_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("genotype.id"), nullable=False, init=False
    )
    genotype: orm.Mapped[Genotype] = orm.relationship()
    fitness: orm.Mapped[int] = orm.mapped_column(nullable=False)
