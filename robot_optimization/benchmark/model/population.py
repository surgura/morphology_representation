import sqlalchemy.orm as orm
from .base import Base
from .individual import Individual
from typing import List
import sqlalchemy.ext.orderinglist


class Population(Base):
    __tablename__ = "population"

    id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    individuals: orm.Mapped[List[Individual]] = orm.relationship(
        order_by=Individual.population_index,
        collection_class=sqlalchemy.ext.orderinglist.ordering_list("population_index"),
    )
