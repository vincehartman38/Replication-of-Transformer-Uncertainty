import argparse
from typing import List, Tuple, Union
from itertools import chain
import torch
import random
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)
from utils import entropy, nucleus_sampling, create_entropy_histogram
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
    token_entropies: dict,
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

    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_scores=True,
    )
    for seq_idx in range(model_output.sequences.shape[0]):
        for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
            beam_idx = model_output.beam_indices[seq_idx][idx]
            selected_beam_probs = torch.exp(model_output.scores[idx][beam_idx])
            # perform nucleus sampling
            beam_probs_nucleus = nucleus_sampling(selected_beam_probs)
            result = entropy(beam_probs_nucleus)
            if output_token_id in inputs.input_ids:
                token_entropies["existing"].append(result)
            else:
                token_entropies["novel"].append(result)


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
        help="choose the number of gnerations steps for the model",
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
    token_entropies = {"existing": [], "novel": []}
    count = 0
    for x in data_keys:
        selected_data = data.data_by_id[x]
        source = [selected_data["document"]]
        generate_entropies(model, tokenizer, source, token_entropies)
        count += len(list(chain(*token_entropies.values())))
        print("Progress: {} tokens completed of {}".format(count, args.steps))
        if count > args.steps:
            print("Completed Entropy Generation Steps")
            break
    create_entropy_histogram(token_entropies, args.model, count)
