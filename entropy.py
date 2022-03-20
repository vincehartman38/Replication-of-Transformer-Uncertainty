import argparse
import torch
import numpy as np
from utils import load_model_and_tokenizer
from figures import create_bigram_histogram, create_position_boxplot
from storage import get_summaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to product Replication Figures 1 and 2"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "pegasus-cnn_dailymail",
            "pegasus-xsum",
            "bart-large-cnn",
            "bart-large-xsum",
        ],
        help="choose the model",
    )

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    summaries = get_summaries(args.model, model.config.name_or_path)

    # Existing Bigram means the bigram just generated occurs in the input document,
    # while a NovelBigram is an organic model generation.
    bigram_entropies = {"existing": [], "novel": []}
    # Prediction entropy values by relative sentence positions.
    # For example, 0.0 indicates the first 10% of tokens in a sentence, and 0.9 is the last 10% of tokens.
    position_entropies = {key: [] for key in np.round(np.linspace(0, 0.9, 10), 1)}

    for key, value in summaries.items():
        entropies = value["metadata"]["entropy"]
        bigram_source = value["metadata"]["bigrams_in_input"]
        sentence_position = value["metadata"]["sentence_position"]
        for ind, e in enumerate(entropies):
            if bigram_source[ind]:
                bigram_entropies["existing"].append(e)
            else:
                bigram_entropies["novel"].append(e)
            position_entropies[sentence_position[ind]].append(e)

    create_bigram_histogram(bigram_entropies, args.model)
    create_position_boxplot(position_entropies, args.model)
