import argparse
from typing import List, Union
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
from dataset import XSumDataset, CNNDataset
from utils import entropy, load_model_and_tokenizer
from datasets import load_dataset
from storage import store_model_summaries
from figures import create_bigram_histogram, create_position_boxplot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_entropies(
    model: Union[BartForConditionalGeneration, PegasusForConditionalGeneration],
    tokenizer: Union[BartTokenizer, PegasusTokenizer],
    docs_to_summarize: List[str],
    bigram_entropies: dict,
    position_entropies: dict,
    num_beams: int = 4,
    max_length: int = 1024,
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
        max_length=max_length,
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

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]
    token_metadata = {
        "token_ids": [],
        "tokens": [],
        "entropy": [],
        "bigrams": [],
        "tokens_in_input": [],
        "bigrams_in_input": [],
        "sentence_position": [],
    }
    for seq_idx in range(model_output.sequences.shape[0]):
        previous_token = model_output.sequences[seq_idx][0]
        all_tokens = model_output.sequences[seq_idx][1:-1]
        # split tokens into sentences. Token 107 is a period
        indices = [i for i, x in enumerate(all_tokens) if x == 107]
        init = 0
        for ind in indices:
            sequence = all_tokens[init : ind + 1]
            for idx, output_token_id in enumerate(sequence):
                loc = init + idx
                bigram = (previous_token, output_token_id)
                previous_token = output_token_id
                beam_idx = model_output.beam_indices[seq_idx][loc]
                selected_beam_probs = torch.exp(model_output.scores[loc][beam_idx])
                result = entropy(selected_beam_probs)
                token_metadata["token_ids"].append(output_token_id.item())
                token_metadata["tokens"].append(tokenizer.decode(output_token_id))
                token_metadata["entropy"].append(result)
                if bigram in input_bigrams:
                    bigram_entropies["existing"].append(result)
                else:
                    bigram_entropies["novel"].append(result)
                # place entropies into dictionary bucket based on position

                position = idx / len(sequence)
                rounded_position = math.floor(position * 10) / 10.0
                position_entropies[rounded_position].append(result)
                token_metadata["bigrams"].append(tuple([x.item() for x in bigram]))
                token_metadata["tokens_in_input"].append(
                    output_token_id in input_token_ids
                )
                token_metadata["bigrams_in_input"].append(bigram in input_bigrams)
                token_metadata["sentence_position"].append(rounded_position)
            init = ind + 1

    return generated_summaries, token_metadata


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

    model, tokenizer = load_model_and_tokenizer(args.model)
    if args.model.split("-")[-1] == "xsum":
        data = load_dataset("xsum")
        data = XSumDataset(data["test"])
        max_length = 512
    else:
        data = load_dataset("ccdv/cnn_dailymail", "3.0.0")
        data = CNNDataset(data["test"])
        max_length = 1024
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
        source_document = [selected_data["document"]]
        generated_summary, token_metadata = generate_entropies(
            model,
            tokenizer,
            source_document,
            bigram_entropies,
            position_entropies,
            max_length=max_length,
        )
        count = len(list(chain(*bigram_entropies.values())))

        store_model_summaries(
            args.model,
            model.config.name_or_path,
            model.config.to_dict(),
            {x: generated_summary},
            {x: token_metadata},
        )

        print("Progress: {} tokens completed of {}".format(count, args.steps))
        if count > args.steps:
            print("Completed Entropy Generation Steps")
            break

    create_bigram_histogram(bigram_entropies, args.model)
    create_position_boxplot(position_entropies, args.model)
