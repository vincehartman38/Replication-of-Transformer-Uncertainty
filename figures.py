import matplotlib.pyplot as plt
import numpy as np
import statistics
from pathlib import Path
import time
from itertools import chain

# save histograms to results folder
results_path = "./results"
Path(results_path).mkdir(parents=True, exist_ok=True)


def create_bigram_histogram(data: dict, m_name: str):
    """
    Creates a histogram of the distribution of existing and novel
    bigrams for the generated model. Saves the histogram as a jpeg
    in the results folder

    Args:
        data: dictionary of the entropy values in the format
        {"existing": [], "novel": []}
        m_name: the transformer model
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    count = len(list(chain(*data.values())))
    bins = np.linspace(0, 5, 16)
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
    plt.savefig(results_path + "/" + m_name + "_hisotgram_" + timestr + ".jpeg")
    plt.show()
    plt.close()


def create_position_boxplot(data: dict, m_name: str):
    """
    Creates a vertical boxplot of the predicted entropy values by
    relative sentenence position. Saves the boxplot as a jpeg
    in the results folder

    Args:
        data: dictionary of the sentence position values in the format
        {0.0: [] 0.1: [], 0.2: [], 0.3:[], 0.4:[], 0.5:[],
        0.6:[], 0.7:[], 0.8:, 0.9:[]}
        m_name: the transformer model
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    keys = list(data.keys())
    print(keys)
    keys.sort()  # put keys in order
    print(keys)
    count = len(list(chain(*data.values())))
    plt.figure()
    plt.hold = True
    boxes = []
    median_vals = []
    for k in keys:
        boxes.append(data[k])
        median_vals.append(statistics.median(data[k]))
    norm = plt.Normalize(min(median_vals), max(median_vals))
    colors = plt.cm.coolwarm(norm(median_vals))
    bplot = plt.boxplot(boxes, sym="", vert=True, labels=keys, patch_artist=True)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    plt.title(m_name.title() + " of " + str(count) + " Generation Steps")
    plt.xlabel("Relaive Position")
    plt.ylabel("Entropy")
    plt.savefig(results_path + "/" + m_name + "_boxplot_" + timestr + ".jpeg")
    plt.show()
    plt.close()


def create_syntactic_boxplot(data: dict, m_name: str):
    """
    Creates a vertical boxplot of the entropy change by snytactic
    distance between each token. Saves the boxplot as a jpeg
    in the results folder

    Args:
        data: dictionary of the syntactic distance in the format
        {0: [] 1: [], 2: [], 3:[], 4:[], 5:[]}
        m_name: the transformer model
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    keys = list(data.keys())
    keys.sort()
    plt.figure()
    plt.hold = True
    boxes = []
    median_vals = []
    for k in keys:
        boxes.append(data[k])
        if data[k]:
            median_vals.append(statistics.median(data[k]))
        else:
            median_vals.append(0)
    labels = keys
    labels[-1] = str(labels[-1]) + "+"
    plt.boxplot(boxes, sym="", vert=True, labels=labels)
    plt.axhline(y=0, color="black", linestyle="--")
    plt.title(m_name.split("-")[-1].title())
    plt.xlabel("Syntactic Distance")
    plt.ylabel("Entropy Change (%)")
    plt.savefig(results_path + "/" + m_name + "_syntactic_" + timestr + ".jpeg")
    plt.show()
    plt.close()


def create_attention_graph(data: dict):
    """
    Creates a line graph of the correlation of cross attention
    entropy and prediciton entropy. Saves the line graph as a jpeg
    in the results folder

    Args:
        data: dictionary of the cross entropy in the format
        {bart-cnn:{0.0: [], 0.5: [], ... ,4.5: [] }, ...}
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig = plt.figure(figsize=(8, 4))
    fig.show()
    ax = fig.add_subplot(111)
    colors = {
        "pegasus-cnn_dailymail": "cornflowerblue",
        "pegasus-xsum": "darkorange",
        "bart-large-xsum": "firebrick",
        "bart-large-cnn": "limegreen",
    }
    for m_name, buckets in data.items():
        keys = list(buckets.keys())
        keys.sort()
        mean_vals = []
        for k in buckets:
            if buckets[k]:
                mean_vals.append(statistics.mean(buckets[k]))
            elif k == 0:
                mean_vals.append(0)
            else:
                mean_vals.append(mean_vals[-1])
        ax.plot(
            keys,
            mean_vals,
            c=colors[m_name],
            marker="x",
            ls="--",
            label=m_name,
            fillstyle="none",
        )
    plt.xticks(keys)
    plt.legend(loc=2)
    plt.title("Correlation Between Attention & Entropy")
    plt.xlabel("Prediction Entropy")
    plt.ylabel("Attention Entropy")
    plt.savefig(results_path + "/attention_and_entropy_" + timestr + ".jpeg")
    plt.show()
    plt.close()
