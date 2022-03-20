from typing import Tuple, Union
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
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


def get_change(current, previous):
    """ "
    Returns percent change between two numbers

    Args:
        current: float number
        previous: float number

    Returns:
        percent change between the two numbers (float)
    """
    if current == previous:
        return 100.0
    try:
        return (current - previous) / previous * 100.0
    except ZeroDivisionError:
        return 0
