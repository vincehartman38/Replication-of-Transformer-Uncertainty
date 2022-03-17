import argparse
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import random
import transformers
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)
import datasets
from datasets import load_dataset

from dataset import XSumDataset, CNNDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    print(model_output)
    for seq_idx in range(model_output.sequences.shape[0]):
        for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
            beam_idx = model_output.beam_indices[seq_idx][idx]
            selected_beam_probs = torch.exp(model_output.scores[idx][beam_idx])

            print(output_token_id, entropy(selected_beam_probs))


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

    args = parser.parse_args()

    model, tokenizer = load_summarization_model_and_tokenizer(args.model)
    if args.model.split("-")[-1] == "xsum":
        dataset = load_dataset("xsum")
        data = XSumDataset(dataset["test"])
    else:
        dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0")
        data = CNNDataset(dataset["test"])

    random_value = random.choice(list(data))
    selected_data = [random_value["document"]]
    # print(selected_data)

    generate_entropies(model, tokenizer, selected_data)
