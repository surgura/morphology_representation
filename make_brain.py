from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic,
)
import math
from robot_optimization.generic.brain_parameters import BrainParameters
from typing import Tuple


def make_brain(
    cpg_network_structure: CpgNetworkStructure, params: Tuple[float, ...]
) -> BrainCpgNetworkStatic:
    initial_state = cpg_network_structure.make_uniform_state(0.5 * math.pi / 2.0)
    weight_matrix = cpg_network_structure.make_connection_weights_matrix_from_params(
        list(params)
    )
    dof_ranges = cpg_network_structure.make_uniform_dof_ranges(1.0)
    return BrainCpgNetworkStatic(
        initial_state,
        cpg_network_structure.num_cpgs,
        weight_matrix,
        dof_ranges,
    )
