import argparse
from typing import List, Tuple, Union
from itertools import chain
import numpy as np
import math
import torch
import random
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)
from utils import entropy, create_bigram_histogram, create_position_boxplot
from datasets import load_dataset
from dataset import XSumDataset, CNNDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_summarization_model_and_tokenizer(
    m_name: str,
) -> Tuple[
    Union[BartForConditionalGeneration, PegasusForConditionalGeneration],
    Union[BartTokenizer, PegasusTokenizer],
]:
    """
    Load summary generation model and move to GPU, if possible.

    Args:
        m_name: model name to load tokenizer and model from Hugging Face
    Returns:
        (model, tokenizer)
    """
    if m_name.split("-")[0] == "pegasus":
        tokenizer = PegasusTokenizer.from_pretrained("google/" + m_name)
        model = PegasusForConditionalGeneration.from_pretrained("google/" + m_name)
    elif m_name.split("-")[0] == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/" + m_name)
        model = BartForConditionalGeneration.from_pretrained("facebook/" + m_name)
    model.to(device)

    return model, tokenizer


def generate_entropies(
    model: Union[BartForConditionalGeneration, PegasusForConditionalGeneration],
    tokenizer: Union[BartTokenizer, PegasusTokenizer],
    docs_to_summarize: List[str],
    bigram_entropies: dict,
    position_entropies: dict,
    num_beams: int = 4,
):
    """
    Given a model and tokenizer,

    1. Tokenize text
    2. Run inference on model
    3. Decode tokens using the tokenizer

    Args:
        model: model to run inference on
        tokenizer: tokenizer corresponding to model
        docs_to_summarize: documents to summarize
        bigram_entropies: dictonary of entropies for existing & novel
        position_entropies: dictoary of entropies for sentence position
        num_beams: number of beams for beam search

    Returns:
        entropy
    """

    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    input_token_ids = inputs.input_ids.to(device)
    input_bigrams = [b for l in input_token_ids for b in zip(l[:-1], l[1:])]
    # top_p is nucleus sampling based on Holtzman et al.
    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_scores=True,
        top_p=0.95,
    )
    for seq_idx in range(model_output.sequences.shape[0]):
        previous_token = model_output.sequences[seq_idx][0]
        for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
            bigram = (previous_token, output_token_id)
            previous_token = output_token_id
            beam_idx = model_output.beam_indices[seq_idx][idx]
            selected_beam_probs = torch.exp(model_output.scores[idx][beam_idx])
            result = entropy(selected_beam_probs)
            if bigram in input_bigrams:
                bigram_entropies["existing"].append(result)
            else:
                bigram_entropies["novel"].append(result)
            # place entropies into dictionary bucket based on position
            position = idx / len(model_output.sequences[seq_idx])
            rounded_position = math.floor(position * 10) / 10.0
            position_entropies[rounded_position].append(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate next token entropy"
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

    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="choose the number of generations steps for the model",
    )

    args = parser.parse_args()

    model, tokenizer = load_summarization_model_and_tokenizer(args.model)
    if args.model.split("-")[-1] == "xsum":
        data = load_dataset("xsum")
        data = XSumDataset(data["test"])
    else:
        data = load_dataset("ccdv/cnn_dailymail", "3.0.0")
        data = CNNDataset(data["test"])

    data_keys = data.keys
    random.Random(42).shuffle(data_keys)
    # Existing Bigram means the bigram just generated occurs in the input document,
    # while a NovelBigram is an organic model generation.
    bigram_entropies = {"existing": [], "novel": []}
    # Prediction entropy values by relative sentence positions.
    # For example, 0.0 indicates the first 10% of tokens in a sentence, and 0.9 is the last 10% of tokens.
    position_entropies = {key: [] for key in np.round(np.linspace(0, 0.9, 10), 1)}
    count = 0
    for x in data_keys:
        selected_data = data.data_by_id[x]
        source = [selected_data["document"]]
        generate_entropies(
            model, tokenizer, source, bigram_entropies, position_entropies
        )
        count = len(list(chain(*bigram_entropies.values())))
        print("Progress: {} tokens completed of {}".format(count, args.steps))
        if count > args.steps:
            print("Completed Entropy Generation Steps")
            break
    create_bigram_histogram(bigram_entropies, args.model)
    create_position_boxplot(position_entropies, args.model)
