"""Visualize and run a modular robot using Mujoco."""

import math
from random import Random

from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbourRandom
from revolve2.core.physics.running import (
    ActorControl,
    Batch,
    Environment,
    EnvironmentController,
    PosedActor,
)
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains, modular_robots
import asyncio


# This is exactly the same as the revolve class `revolve2.core.physics.environment_actor_controller.EnvironmentActorController`
class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controller: ActorController

    def __init__(self, actor_controller: ActorController) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for the single actor in the environment.
        """
        self.actor_controller = actor_controller

    def control(self, dt: float, actor_control: ActorControl) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        """
        self.actor_controller.step(dt)
        actor_control.set_dof_targets(0, self.actor_controller.get_dof_targets())


class Simulator:
    """
    Simulator setup.

    Simulates using Mujoco.
    Defines a control function that steps the controller and applies the degrees of freedom the controller provides.
    """

    async def simulate(self, robot: ModularRobot, control_frequency: float) -> None:
        """
        Simulate a robot.

        :param robot: The robot to simulate.
        :param control_frequency: Control frequency for the simulator.
        """
        batch = Batch(
            simulation_time=1000000,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
        )

        actor, controller = robot.make_actor_and_controller()
        bounding_box = actor.calc_aabb()

        env = Environment(EnvironmentActorController(controller))
        env.static_geometries.extend(terrains.flat().static_geometry)
        env.actors.append(
            PosedActor(
                actor,
                Vector3(
                    [
                        0.0,
                        0.0,
                        bounding_box.size.z / 2.0 - bounding_box.offset.z,
                    ]
                ),
                Quaternion(),
                [0.0 for _ in controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner()
        await runner.run_batch(batch)


async def a_simulate(body: Body) -> None:
    rng = Random()
    rng.seed(5)

    brain = BrainCpgNetworkNeighbourRandom(rng)
    robot = ModularRobot(body, brain)

    sim = Simulator()

    await sim.simulate(robot, 60)


def simulate(body: Body) -> None:
    asyncio.run(a_simulate(body))
