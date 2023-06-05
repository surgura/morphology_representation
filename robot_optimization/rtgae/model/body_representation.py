"""BodyRepresentation for a modular robot body and brain."""

from typing import List

import multineat
import numpy as np
import sqlalchemy
import sqlalchemy.orm as orm
import torch
from typing_extensions import Self

import config
from revolve2.core.modular_robot import Body
from robot_rgt import tree_to_body
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from tree import GraphAdjform

from .base import Base


class BodyRepresentation(Base):
    """Database model for the body representation."""

    __tablename__ = "body_representation"

    id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    serialized_body: orm.Mapped[str] = orm.mapped_column(init=False, nullable=False)
    body: List[float]

    def __post_init__(self) -> None:
        self.serialized_body = ";".join([str(p) for p in self.body])

    @sqlalchemy.orm.reconstructor
    def init_on_load(self):
        self.body = [float(s) for s in self.serialized_body.split(";")]

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        representation_size: int,
    ) -> Self:
        """
        Create a random representation.

        :param rng: Random number generator.
        :param representation_size: Number of parameters for the body representation.
        :returns: The created genotype.
        """
        body = (rng.random(representation_size) * 2.0 - 1.0).tolist()

        return BodyRepresentation(body)

    @staticmethod
    def bounce(val: float) -> float:
        """
        Bounce a value between 0 and 1, inclusive.

        :param val: The value to bounce.
        :returns: The bounced result.
        """
        val = abs(val) % 2
        if val > 1:
            return 2 - val
        else:
            return val

    def mutate(
        self,
        rng: np.random.Generator,
    ) -> Self:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        pertubations = rng.normal(0.0, config.OPTRTGAE_MUTATE_SIGMA, len(self.body))
        mutated_body = [self.bounce(float(p)) for p in (pertubations + self.body)]

        return BodyRepresentation(
            mutated_body,
        )

    @staticmethod
    def uniform_crossover(
        parent1: List[float], parent2: List[float], rng: np.random.Generator
    ) -> List[float]:
        assert len(parent1) == len(parent2)

        return [p1 if rng.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

    @classmethod
    def crossover(
        cls,
        parent1: Self,
        parent2: Self,
        rng: np.random.Generator,
    ) -> Self:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        return BodyRepresentation(
            cls.uniform_crossover(parent1.body, parent2.body, rng),
        )

    def develop(self, body_model: TreeGrammarAutoEncoder) -> Body:
        """
        Develop the genotype into a modular robot.

        :param genotype: The genotype to create the robot from.
        :returns: The created robot.
        """
        nodes, adj, _ = body_model.decode(
            torch.tensor(self.body), max_size=config.MODEL_MAX_MODULES_INCL_EMPTY
        )
        return tree_to_body(GraphAdjform(nodes, adj))
