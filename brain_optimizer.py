import numpy as np
from revolve2.core.modular_robot import Body
from robot_to_actor_cpg import robot_to_actor_cpg
import cma
import config
import logging
from evaluator import Evaluator
from revolve2.core.modular_robot import ModularRobot
from make_brain import make_brain
from typing import Tuple


def optimize(
    evaluator: Evaluator, rng: np.random.Generator, body: Body
) -> Tuple[float, ...]:
    logging.info("Starting brain optimization.")

    _, cpg_network_structure = robot_to_actor_cpg(body)
    if cpg_network_structure.num_connections == 0:
        return tuple()
    initial_brain = max(cpg_network_structure.num_connections, 2) * [0.0]

    options = cma.CMAOptions()
    options.set("seed", rng.integers(0, 2**15))
    options.set("bounds", [-1.0, 1.0])
    opt = cma.CMAEvolutionStrategy(
        initial_brain, config.ROBOPT_BRAIN_INITIAL_STD, options
    )

    gen = 0
    while gen < config.ROBOPT_BRAIN_NUM_GENERATIONS:
        logging.info(
            f"Brain opt gen {gen + 1} / {config.ROBOPT_BRAIN_NUM_GENERATIONS}."
        )

        solutions = [tuple(float(p) for p in params) for params in opt.ask()]
        robots = [
            ModularRobot(
                body=body,
                brain=make_brain(
                    cpg_network_structure,
                    solution[0 : cpg_network_structure.num_connections],
                ),
            )
            for solution in solutions
        ]
        fitnesses = [-1.0 * x for x in evaluator.evaluate(robots)]
        opt.tell(solutions, fitnesses)
        logging.info(f"{opt.result.xbest=} {opt.result.fbest=}")

        gen += 1

    logging.info("Brain optimization done.")

    return tuple(
        float(x) for x in opt.result.xbest[0 : cpg_network_structure.num_connections]
    )
