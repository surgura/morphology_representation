import numpy as np
from revolve2.core.modular_robot import Body
import robot_optimization.rtgae.model as model
from robot_to_actor_cpg import robot_to_actor_cpg


def optimize(rng: np.random.Generator, body: Body) -> model.BrainParameters:
    actor, cpg_network_structure = robot_to_actor_cpg(body)
    initial_brain = model.BrainParameters(cpg_network_structure.num_connections * [0.5])
    return initial_brain
