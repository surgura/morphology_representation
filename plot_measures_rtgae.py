import logging
import pathlib
import pickle

import matplotlib.pyplot as plt
import pandas

import config
from typing import Tuple, List
import argparse


def plot_measure(df: pandas.DataFrame, measure_name: str, out_file: str) -> None:
    fig = plt.figure()
    ax = plt.subplot(projection="3d")
    ax.set_xlabel("t_dim")
    ax.set_ylabel("r_dim")
    ax.set_zlabel(measure_name)

    surf = ax.plot_trisurf(
        df["t_dim"],
        df["r_dim"],
        df[measure_name],
        cmap=plt.cm.coolwarm_r,
        linewidth=0.2,
    )
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def plot_pairs(
    pairs: List[Tuple[float, float]], out_file: str, xname: str, yname: str
) -> None:
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    args = parser.parse_args()
    experiment_name = args.experiment_name

    all_records_coverage = []
    all_records_stress = []

    for run in range(config.RUNS):
        run_records_coverage = []
        run_records_stress = []
        for t_dim_i in range(len(config.MODEL_T_DIMS)):
            for r_dim_i in range(len(config.MODEL_R_DIMS)):
                t_dim = config.MODEL_T_DIMS[t_dim_i]
                r_dim = config.MODEL_R_DIMS[r_dim_i]
                with open(
                    config.CVGRTGAE_OUT(
                        experiment_name=experiment_name,
                        run=run,
                        t_dim=t_dim,
                        r_dim=r_dim,
                    ),
                    "rb",
                ) as f:
                    coverage = pickle.load(f)
                    assert isinstance(coverage, float)
                    run_records_coverage.append((t_dim, r_dim, coverage))
                    all_records_coverage.append((run, t_dim, r_dim, coverage))

                with open(
                    config.STRESSRTGAE_OUT(
                        experiment_name=experiment_name,
                        run=run,
                        t_dim=t_dim,
                        r_dim=r_dim,
                    ),
                    "rb",
                ) as f:
                    measures = pickle.load(f)
                    stress = measures["stress"]
                    assert isinstance(stress, float)
                    run_records_stress.append((t_dim, r_dim, stress))
                    all_records_stress.append((run, t_dim, r_dim, stress))

                    pairs: Tuple[float, float] = measures["dist_pairs"]
                    plot_pairs(
                        pairs,
                        config.PLTMSR_OUT_PAIRS(
                            experiment_name=experiment_name,
                            run=run,
                            t_dim=t_dim,
                            r_dim=r_dim,
                        ),
                        "representation_distance",
                        "solution_distance",
                    )
        df_coverage = pandas.DataFrame.from_records(
            run_records_coverage, columns=["t_dim", "r_dim", "coverage"]
        )
        df_stress = pandas.DataFrame.from_records(
            run_records_stress, columns=["t_dim", "r_dim", "stress"]
        )
        plot_measure(
            df_coverage,
            "coverage",
            config.PLTMSR_OUT_COVERAGE_INDIVIDUAL_RUNS(
                experiment_name=experiment_name, run=run
            ),
        )
        plot_measure(
            df_stress,
            "stress",
            config.PLTMSR_OUT_STRESS_INDIVIDUAL_RUNS(
                experiment_name=experiment_name, run=run
            ),
        )

    df_coverage = pandas.DataFrame.from_records(
        all_records_coverage, columns=["run", "t_dim", "r_dim", "coverage"]
    )
    df_stress = pandas.DataFrame.from_records(
        all_records_stress, columns=["run", "t_dim", "r_dim", "stress"]
    )

    # TODO average multiple runs

    # fig = plt.figure()
    # ax = plt.subplot(projection="3d")
    # ax.set_xlabel("t_dim")
    # ax.set_ylabel("r_dim")
    # ax.set_zlabel("distance_distortion (lower is better)")

    # surf = ax.plot_trisurf(
    #     df["t_dim"],
    #     df["r_dim"],
    #     df["distance_distortion"],
    #     cmap=plt.cm.coolwarm_r,
    #     linewidth=0.2,
    # )
    # pathlib.Path(config.PLTMSR_OUT_COVERAGE_COMBINED_RUNS).parent.mkdir(
    #     parents=True, exist_ok=True
    # )
    # plt.savefig(config.PLTMSR_OUT_COVERAGE_COMBINED_RUNS, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()
