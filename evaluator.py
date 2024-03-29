import asyncio
import math
from typing import List, Optional

from pyrr import Quaternion, Vector3

import config
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics import Terrain
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)
from revolve2.core.physics.running import ActorState, Batch
from revolve2.core.physics.running import Environment as PhysicsEnv
from revolve2.core.physics.running import PosedActor, RecordSettings, Runner
from revolve2.runners.mujoco import LocalRunner
from revolve2.standard_resources import terrains


class Evaluator:
    _runner: Runner
    _terrain: Terrain

    def __init__(self, headless: bool, num_simulators: int) -> None:
        self._runner = LocalRunner(headless=headless, num_simulators=num_simulators)
        self._terrain = terrains.flat()

    def evaluate(
        self,
        robots: List[ModularRobot],
        record_settings: Optional[RecordSettings] = None,
    ) -> List[float]:
        batch = Batch(
            simulation_time=config.ROBOPT_SIMULATION_TIME,
            sampling_frequency=config.ROBOPT_SAMPLING_FREQUENCY,
            control_frequency=config.ROBOPT_CONTROL_FREQUENCY,
        )

        for robot in robots:
            actor, controller = robot.make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            env = PhysicsEnv(EnvironmentActorController(controller))
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
            env.static_geometries.extend(self._terrain.static_geometry)
            batch.environments.append(env)

        batch_results = asyncio.run(
            self._runner.run_batch(batch, record_settings=record_settings)
        )

        fitnesses = [
            self._calculate_fitness(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0],
            )
            for environment_result in batch_results.environment_results
        ]
        return fitnesses

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # distance traveled on the xy plane
        return math.sqrt(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )
