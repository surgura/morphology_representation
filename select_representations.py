import logging
import pathlib
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np

import config


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

        out_dir = config.SREP_OUT(run)
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        with open(out_dir, "wb") as f:
            pickle.dump({"best": best, "worst": worst}, f)


if __name__ == "__main__":
    main()
