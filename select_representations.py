import logging
import pickle
import config
from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class Measure:
    measure: int
    t_dim: int
    r_dim: int


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    for run in range(config.RUNS):
        measures: List[Measure] = []
        for t_dim_i in range(len(config.MODEL_T_DIMS)):
            for r_dim_i in range(len(config.MODEL_R_DIMS)):
                t_dim = config.MODEL_T_DIMS[t_dim_i]
                r_dim = config.MODEL_R_DIMS[r_dim_i]
                with open(config.MLOC_OUT(run, t_dim, r_dim), "rb") as f:
                    measure = pickle.load(f)
                    assert isinstance(measure, float)
                    measures.append(Measure(measure, t_dim, r_dim))
        sorted = np.argsort([m.measure for m in measures])
        best = measures[sorted[0]]
        worst = measures[sorted[-1]]
        print(best)
        print(worst)


if __name__ == "__main__":
    main()
