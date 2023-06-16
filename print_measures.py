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
        for t_dim in config.MODEL_T_DIMS:
            for r_dim in config.MODEL_R_DIMS:
                for margin in config.TRAIN_DD_MARGINS:
                    for gain in config.TRAIN_DD_TRIPLET_FACTORS:
                        with open(
                            config.CVGRTGAE_OUT(
                                experiment_name=experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                                margin=margin,
                                gain=gain,
                            ),
                            "rb",
                        ) as fcvg, open(
                            config.STRESSRTGAE_OUT(
                                experiment_name=experiment_name,
                                run=run,
                                t_dim=t_dim,
                                r_dim=r_dim,
                                margin=margin,
                                gain=gain,
                            ),
                            "rb",
                        ) as fstress:
                            coverage = pickle.load(fcvg)
                            stress = pickle.load(fstress)["stress"]
                            print(
                                f"{coverage=} {stress=} {t_dim=} {r_dim=} {margin=} {gain=} {run=}"
                            )


if __name__ == "__main__":
    main()
