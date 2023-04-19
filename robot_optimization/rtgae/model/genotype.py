"""Genotype for a modular robot body and brain."""

import multineat
import sqlalchemy
from revolve2.core.modular_robot import ModularRobot
from revolve2.genotypes.cppnwin import crossover_v1, mutate_v1
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
    develop_v1 as brain_develop,
)
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
    random_v1 as brain_random,
)
from typing import List
from typing_extensions import Self
import numpy as np
from .base import Base
from revolve2.genotypes.cppnwin import Genotype as CPNNWINGenotype
import sqlalchemy.orm as orm
from rtgae.recursive_tree_grammar_auto_encoder import TreeGrammarAutoEncoder
from tree import GraphAdjform
from robot_rgt import tree_to_body
import torch
import config


def _make_multineat_params() -> multineat.Parameters:
    multineat_params = multineat.Parameters()

    multineat_params.MutateRemLinkProb = 0.02
    multineat_params.RecurrentProb = 0.0
    multineat_params.OverallMutationRate = 0.15
    multineat_params.MutateAddLinkProb = 0.08
    multineat_params.MutateAddNeuronProb = 0.01
    multineat_params.MutateWeightsProb = 0.90
    multineat_params.MaxWeight = 8.0
    multineat_params.WeightMutationMaxPower = 0.2
    multineat_params.WeightReplacementMaxPower = 1.0
    multineat_params.MutateActivationAProb = 0.0
    multineat_params.ActivationAMutationMaxPower = 0.5
    multineat_params.MinActivationA = 0.05
    multineat_params.MaxActivationA = 6.0

    multineat_params.MutateNeuronActivationTypeProb = 0.03

    multineat_params.MutateOutputActivationFunction = False

    multineat_params.ActivationFunction_SignedSigmoid_Prob = 0.0
    multineat_params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    multineat_params.ActivationFunction_Tanh_Prob = 1.0
    multineat_params.ActivationFunction_TanhCubic_Prob = 0.0
    multineat_params.ActivationFunction_SignedStep_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedStep_Prob = 0.0
    multineat_params.ActivationFunction_SignedGauss_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedGauss_Prob = 0.0
    multineat_params.ActivationFunction_Abs_Prob = 0.0
    multineat_params.ActivationFunction_SignedSine_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedSine_Prob = 0.0
    multineat_params.ActivationFunction_Linear_Prob = 1.0

    multineat_params.MutateNeuronTraitsProb = 0.0
    multineat_params.MutateLinkTraitsProb = 0.0

    multineat_params.AllowLoops = False

    return multineat_params


_MULTINEAT_PARAMS = _make_multineat_params()


class Genotype(Base):
    """Database model for the genotype."""

    __tablename__ = "genotype"

    id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    serialized_body: orm.Mapped[str] = orm.mapped_column(init=False, nullable=False)
    serialized_brain: orm.Mapped[str] = orm.mapped_column(init=False, nullable=False)

    body: List[float]
    brain: CPNNWINGenotype

    def __post_init__(self) -> None:
        self.serialized_body = ";".join([str(p) for p in self.body])
        self.serialized_brain = self.brain.genotype.Serialize()

    @sqlalchemy.orm.reconstructor
    def init_on_load(self):
        self.body = [float(s) for s in self.serialized_body.split(";")]
        brain_genotype = multineat.Genome()
        brain_genotype.Deserialize(self.serialized_brain)
        self.brain = CPNNWINGenotype(brain_genotype)

    @classmethod
    def random(
        cls,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
        num_initial_mutations: int,
        body_genotype_size: int,
    ) -> Self:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :param num_initial_mutations: The number of times to mutate to create a random network. See CPPNWIN genotype.
        :returns: The created genotype.
        """
        multineat_rng = cls.__multineat_rng_from_random(rng)

        body = rng.random(body_genotype_size).tolist()

        brain = brain_random(
            innov_db_brain,
            multineat_rng,
            _MULTINEAT_PARAMS,
            multineat.ActivationFunction.SIGNED_SINE,
            num_initial_mutations,
        )

        return Genotype(body, brain)

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
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> Self:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        multineat_rng = self.__multineat_rng_from_random(rng)

        pertubations = rng.normal(0.0, config.OPTRTGAE_MUTATE_SIGMA, len(self.body))
        mutated_body = [self.bounce(float(p)) for p in (pertubations + self.body)]

        return Genotype(
            mutated_body,
            mutate_v1(self.brain, _MULTINEAT_PARAMS, innov_db_brain, multineat_rng),
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
        multineat_rng = cls.__multineat_rng_from_random(rng)

        return Genotype(
            cls.uniform_crossover(parent1.body, parent2.body, rng),
            crossover_v1(
                parent1.brain,
                parent2.brain,
                _MULTINEAT_PARAMS,
                multineat_rng,
                False,
                False,
            ),
        )

    def develop(self, body_model: TreeGrammarAutoEncoder) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :param genotype: The genotype to create the robot from.
        :returns: The created robot.
        """
        nodes, adj, _ = body_model.decode(torch.tensor(self.body), max_size=32)
        body = tree_to_body(GraphAdjform(nodes, adj))
        brain = brain_develop(self.brain, body)
        return ModularRobot(body, brain)

    @staticmethod
    def __multineat_rng_from_random(rng: np.random.Generator) -> multineat.RNG:
        multineat_rng = multineat.RNG()
        multineat_rng.Seed(rng.integers(0, 2**31))
        return multineat_rng