from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    ForwardRef,
    Generic,
    Type,
    TypeVar,
)

import sqlalchemy
import sqlalchemy.orm as orm
from typing_extensions import Self

from .has_id import HasId
from .init_subclass_get_generic_args import init_subclass_get_generic_args

TBodyGenotype = TypeVar("TBodyGenotype")
TBrainParameters = TypeVar("TBrainParameters")


class Individual(
    HasId, orm.MappedAsDataclass, Generic[TBodyGenotype, TBrainParameters]
):
    # -------------------------------------
    # Class members interesting to the user
    # -------------------------------------
    if TYPE_CHECKING:
        population_id: orm.Mapped[int] = orm.mapped_column(nullable=False, init=False)
        population_index: orm.Mapped[int] = orm.mapped_column(
            nullable=False, init=False
        )
        genotype_id: orm.Mapped[int] = orm.mapped_column(nullable=False, init=False)
        genotype: orm.Mapped[TBodyGenotype] = orm.relationship()
        fitness: orm.Mapped[float] = orm.mapped_column(nullable=False)
        brain_parameters_id: orm.Mapped[int] = orm.mapped_column(
            nullable=False, init=False
        )
        brain_parameters: orm.Mapped[TBrainParameters] = orm.relationship()

    # ----------------------
    # Implementation details
    # ----------------------
    else:

        @orm.declared_attr
        def population_id(cls) -> orm.Mapped[int]:
            return cls.__population_id_impl()

        @orm.declared_attr
        def population_index(cls) -> orm.Mapped[int]:
            return cls.__population_index_impl()

        @orm.declared_attr
        def genotype_id(cls) -> orm.Mapped[int]:
            return cls.__genotype_id_impl()

        @orm.declared_attr
        def genotype(cls) -> orm.Mapped[TBodyGenotype]:
            return cls.__genotype_impl()

        @orm.declared_attr
        def fitness(cls) -> orm.Mapped[float]:
            return cls.__fitness_impl()

        @orm.declared_attr
        def brain_parameters_id(cls) -> orm.Mapped[int]:
            return cls.__brain_parameters_id_impl()

        @orm.declared_attr
        def brain_parameters(cls) -> orm.Mapped[TBrainParameters]:
            return cls.__brain_parameters_impl()

    __type_tbodygenotype: ClassVar[Type[TBodyGenotype]]  # type: ignore[misc]
    __type_tbrainparameters: ClassVar[Type[TBrainParameters]]  # type: ignore[misc]
    __population_table: ClassVar[str]

    def __init_subclass__(
        cls: Type[Self], population_table: str, **kwargs: Dict[str, Any]
    ) -> None:
        generic_types = init_subclass_get_generic_args(cls, Individual)
        assert len(generic_types) == 2
        cls.__type_tbodygenotype = generic_types[0]
        cls.__type_tbrainparameters = generic_types[1]
        assert not isinstance(
            cls.__type_tbodygenotype, ForwardRef
        ), "TBodyGenotype generic argument cannot be a forward reference."
        assert not isinstance(
            cls.__type_tbrainparameters, ForwardRef
        ), "TBrainParameters generic argument cannot be a forward reference."

        cls.__population_table = population_table
        assert isinstance(
            cls.__population_table, str
        ), "population_table argument must be a string."

        super().__init_subclass__(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def __population_id_impl(cls) -> orm.Mapped[int]:
        return orm.mapped_column(
            sqlalchemy.ForeignKey(f"{cls.__population_table}.id"),
            nullable=False,
            init=False,
        )

    @classmethod
    def __population_index_impl(cls) -> orm.Mapped[int]:
        return orm.mapped_column(nullable=False, init=False)

    @classmethod
    def __genotype_id_impl(cls) -> orm.Mapped[int]:
        return orm.mapped_column(
            sqlalchemy.ForeignKey(f"{cls.__type_tbodygenotype.__tablename__}.id"),
            nullable=False,
            init=False,
        )

    @classmethod
    def __genotype_impl(cls) -> orm.Mapped[TBodyGenotype]:
        return orm.relationship(cls.__type_tbodygenotype)

    @classmethod
    def __fitness_impl(cls) -> orm.Mapped[float]:
        return orm.mapped_column(nullable=False)

    @classmethod
    def __brain_parameters_id_impl(cls) -> orm.Mapped[int]:
        return orm.mapped_column(
            sqlalchemy.ForeignKey(f"{cls.__type_tbrainparameters.__tablename__}.id"),
            nullable=False,
            init=False,
        )

    @classmethod
    def __brain_parameters_impl(cls) -> orm.Mapped[TBrainParameters]:
        return orm.relationship(cls.__type_tbrainparameters)
