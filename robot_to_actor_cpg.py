from revolve2.core.modular_robot import Body
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.physics.actor import Actor
from revolve2.core.modular_robot.brains import (
    make_cpg_network_structure_neighbor,
)
from typing import Tuple


def robot_to_actor_cpg(body: Body) -> Tuple[Actor, CpgNetworkStructure]:
    """
    Convert a body to an actor and get it's corresponding cpg network structure.

    :param body: The body to convert.
    :returns: A tuple of the actor and cpg network structure.
    """
    actor, dof_ids = body.to_actor()
    id_to_hinge = {
        active_hinge.id: active_hinge for active_hinge in body.find_active_hinges()
    }
    active_hinges = [id_to_hinge[dof_id] for dof_id in dof_ids]
    cpg_network_structure = make_cpg_network_structure_neighbor(active_hinges)

    return actor, cpg_network_structure
