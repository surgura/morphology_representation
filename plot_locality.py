import logging
import config
import pickle
import pandas
import matplotlib.pyplot as plt
import pathlib


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    all_records = []

    for run in range(config.RUNS):
        run_records = []

        for t_dim_i in range(len(config.MODEL_T_DIMS)):
            for r_dim_i in range(len(config.MODEL_R_DIMS)):
                t_dim = config.MODEL_T_DIMS[t_dim_i]
                r_dim = config.MODEL_R_DIMS[r_dim_i]
                with open(config.MLOC_OUT(run, t_dim, r_dim), "rb") as f:
                    measure = pickle.load(f)
                    assert isinstance(measure, float)
                    run_records.append((t_dim, r_dim, measure))
                    all_records.append((run, t_dim, r_dim, measure))

        df = pandas.DataFrame.from_records(
            run_records, columns=["t_dim", "r_dim", "locality"]
        )

        fig = plt.figure()
        ax = plt.subplot(projection="3d")
        ax.set_xlabel("t_dim")
        ax.set_ylabel("r_dim")
        ax.set_zlabel("locality (lower is better)")

        surf = ax.plot_trisurf(
            df["t_dim"],
            df["r_dim"],
            df["locality"],
            cmap=plt.cm.coolwarm_r,
            linewidth=0.2,
        )
        out_dir = config.PLOC_OUT_INDIVIDUAL_RUNS(run)
        pathlib.Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir)

    df = pandas.DataFrame.from_records(
        all_records, columns=["run", "t_dim", "r_dim", "locality"]
    )

    # TODO average multiple runs

    fig = plt.figure()
    ax = plt.subplot(projection="3d")
    ax.set_xlabel("t_dim")
    ax.set_ylabel("r_dim")
    ax.set_zlabel("locality (lower is better)")

    surf = ax.plot_trisurf(
        df["t_dim"], df["r_dim"], df["locality"], cmap=plt.cm.coolwarm_r, linewidth=0.2
    )
    pathlib.Path(config.PLOC_OUT_COMBINED_RUNS).parent.mkdir(
        parents=True, exist_ok=True
    )
    plt.savefig(config.PLOC_OUT_COMBINED_RUNS)


if __name__ == "__main__":
    main()
