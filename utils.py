import torch
import matplotlib.pyplot as plt
import numpy as np
import statistics
from pathlib import Path

# save histograms to results folder
results_path = "./results"
Path(results_path).mkdir(parents=True, exist_ok=True)


def entropy(p_dist: torch.Tensor) -> float:
    """ "
    Calculates Shannon entropy for a probability distribution

    Args:
        p_dist: probability distribution (torch.Tensor)

    Returns:
        entropy (float)
    """
    # add epsilon because log(0) = nan
    p_dist = p_dist.view(-1) + 1e-12
    return -torch.mul(p_dist, p_dist.log()).sum(0).item()


def nucleus_sampling(probs: torch.Tensor) -> torch.Tensor:
    """ "
    Performs nucleus sampling based on Holtzman et al.

    Args:
        probs: probability distribution (torch.Tensor)

    Returns:
        probability distribution rebalanced
    """
    sorted_probs = sorted(probs, reverse=True)
    beam_probs_nucleus = torch.zeros(len(probs), dtype=torch.float64)
    nuc_sum = 0
    p = 0.95
    for i, val in enumerate(sorted_probs):
        if nuc_sum < p:
            nuc_sum += val
            beam_probs_nucleus[i] = val / p
        else:
            break
    return beam_probs_nucleus


# save histogram image
def create_entropy_histogram(data: dict, m_name: str, count: int):
    bins = np.linspace(0, 5, 15)
    existing_median = statistics.median(data["existing"])
    novel_median = statistics.median(data["novel"])

    plt.hist(
        data["existing"],
        bins,
        alpha=0.5,
        color="lightsteelblue",
        edgecolor="cornflowerblue",
        label="Existing Bigrams",
    )
    plt.hist(
        data["novel"],
        bins,
        alpha=0.5,
        color="lightcoral",
        edgecolor="indianred",
        label="Novel Bigrams",
    )
    plt.axvline(x=existing_median, color="blue", linestyle="--")
    plt.axvline(x=novel_median, color="red", linestyle="--")
    plt.legend(loc="upper right")
    plt.title(m_name.title() + " of " + str(count) + " Generation Steps")
    plt.xlabel("Prediction Entropy")
    plt.ylabel("Count")
    plt.savefig(results_path + "/" + m_name + ".jpeg")
    plt.show()
    plt.close()
