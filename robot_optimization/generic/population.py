from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    ForwardRef,
    Generic,
    List,
    Type,
    TypeVar,
)

import sqlalchemy.ext.orderinglist
import sqlalchemy.orm as orm
from typing_extensions import Self

from .has_id import HasId
from .init_subclass_get_generic_args import init_subclass_get_generic_args

TIndividual = TypeVar("TIndividual")


class Population(HasId, orm.MappedAsDataclass, Generic[TIndividual]):
    # -------------------------------------
    # Class members interesting to the user
    # -------------------------------------
    if TYPE_CHECKING:
        individuals: orm.Mapped[List[TIndividual]]

    # ----------------------
    # Implementation details
    # ----------------------
    else:

        @orm.declared_attr
        def individuals(cls) -> orm.Mapped[List[TIndividual]]:
            return cls.__individuals_impl()

    __type_tindividual: ClassVar[Type[TIndividual]]  # type: ignore[misc]

    def __init_subclass__(cls: Type[Self], /, **kwargs: Dict[str, Any]) -> None:
        generic_types = init_subclass_get_generic_args(cls, Population)
        assert len(generic_types) == 1
        cls.__type_tindividual = generic_types[0]
        assert not isinstance(
            cls.__type_tindividual, ForwardRef
        ), "TIndividual generic argument cannot be a forward reference."

        super().__init_subclass__(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def __individuals_impl(cls) -> orm.Mapped[TIndividual]:
        return orm.relationship(
            cls.__type_tindividual,
            order_by=cls.__type_tindividual.population_index,
            collection_class=sqlalchemy.ext.orderinglist.ordering_list(
                "population_index"
            ),
        )
