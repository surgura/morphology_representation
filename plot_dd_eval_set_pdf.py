import matplotlib.pyplot as plt
import pickle
import config
from typing import List, Tuple
import torch


def plot_pdf(run, r_dim):
    N_BINS = 20

    distances = []

    file = config.DDEVSET_OUT(run, r_dim)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    with open(file, "rb") as f:
        pairs = pickle.load(f)
    distances = [torch.norm(pair[0] - pair[1]).numpy() for pair in pairs]
    print(distances)

    plt.hist(
        distances, bins=N_BINS, density=True, alpha=0.75, label="Euclidean distance PDF"
    )
    plt.xlabel("Distance")
    plt.ylabel("Probability Density")
    plt.title(
        f"PDF of Euclidean distances between random vectors (run={run}, r_dim={r_dim})"
    )
    plt.legend()
    plt.show()


def main() -> None:
    plot_pdf(0, r_dim=64)


if __name__ == "__main__":
    main()
