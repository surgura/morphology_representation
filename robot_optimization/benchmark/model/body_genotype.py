"""BodyGenotype class."""

from __future__ import annotations

import multineat
import numpy as np
from .base import Base
from revolve2.core.database import HasId
from revolve2.core.modular_robot import Body
from revolve2.genotypes.cppnwin.modular_robot import BodyGenotype as GenericBodyGenotype


class BodyGenotype(Base, HasId, GenericBodyGenotype):
    """SQLAlchemy model for a genotype for a modular robot body."""

    __tablename__ = "genotype"

    @classmethod
    def random(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> BodyGenotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        body = cls.random_body(innov_db_body, rng)

        return BodyGenotype(body=body.body)

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> BodyGenotype:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        body = self.mutate_body(innov_db_body, rng)

        return BodyGenotype(body=body.body)

    @classmethod
    def crossover(
        cls,
        parent1: BodyGenotype,
        parent2: BodyGenotype,
        rng: np.random.Generator,
    ) -> BodyGenotype:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        body = cls.crossover_body(parent1, parent2, rng)

        return BodyGenotype(body=body.body)

    def develop(self) -> Body:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        return self.develop_body()
