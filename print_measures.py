import logging
import argparse
import config
import pickle


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    args = parser.parse_args()
    experiment_name = args.experiment_name

    for run in range(config.RUNS):
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
                ) as fcvg, open(
                    config.STRESSRTGAE_OUT(
                        experiment_name=experiment_name,
                        run=run,
                        t_dim=t_dim,
                        r_dim=r_dim,
                    ),
                    "rb",
                ) as fstress:
                    coverage = pickle.load(fcvg)
                    stress = pickle.load(fstress)["stress"]
                    print(f"{coverage=} {stress=} {t_dim=} {r_dim=} {run=}")


if __name__ == "__main__":
    main()
