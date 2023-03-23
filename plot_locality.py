import logging
import config
import pickle
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    records = []

    for run in range(config.RUNS):
        for t_dim_i in range(len(config.MODEL_T_DIMS)):
            for r_dim_i in range(len(config.MODEL_R_DIMS)):
                t_dim = config.MODEL_T_DIMS[t_dim_i]
                r_dim = config.MODEL_R_DIMS[r_dim_i]
                with open(config.MLOC_OUT(run, t_dim, r_dim), "rb") as f:
                    measure = pickle.load(f)
                    assert isinstance(measure, float)
                    records.append((run, t_dim, r_dim, measure))

    df = pandas.DataFrame.from_records(
        records, columns=["run", "t_dim", "r_dim", "locality"]
    )

    # TODO multiple runs

    fig = plt.figure()
    ax = plt.subplot(projection="3d")
    ax.set_xlabel("t_dim")
    ax.set_ylabel("r_dim")
    ax.set_zlabel("locality (lower is better)")

    surf = ax.plot_trisurf(
        df["t_dim"], df["r_dim"], df["locality"], cmap=plt.cm.coolwarm_r, linewidth=0.2
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == "__main__":
    main()
