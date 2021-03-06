import json
import os
from typing import Dict, Optional, Any
import unicodedata
import re
from pathlib import Path

STORAGE_DIR = "./data"
Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)


def dataset_dir(dataset: str):
    return f"{STORAGE_DIR}/{dataset}"


def model_path(dataset: str, model: str, ext):
    return f"{dataset_dir(dataset)}/{slugify(model)}-{ext}"


def slugify(value: str, allow_unicode: bool = False):
    """
    For a given string convert slashes, spaces or repeated dashes to single dashes.
    Removes characters that aren't alphanumerics, underscores, or hyphens.
    Converts to lowercase.
    Strip leading and trailing whitespace, dashes, and underscores.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py

    Args:
        value: the string to slugify
        allow_unicode: Convert to ASCII if 'allow_unicode' is False.

    Returns:
        the slugified string
    """
    value = str(value).replace("/", "-")
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def store_model_config(dataset: str, model: str, model_config: dict):
    model_config["sumtool_model"] = slugify(model)
    model_config_json = json.dumps(model_config, sort_keys=True, indent=2)

    dir = f"{dataset_dir(dataset)}/"
    path = model_path(dataset, model, "config.json")

    if not os.path.exists(path):
        os.makedirs(dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(model_config_json)


def store_model_summaries(
    dataset: str,
    model: str,
    model_config: Dict,
    generated_summaries: Dict[str, str],
    metadata: Optional[Dict[str, Any]] = {},
):
    """
    Stores model summaries, indexed by dataset, model name & document id

    If the storage already has a summary for this document id and model config, it will overwrite it.
    Otherwise summaries are appended to the storage.

    Ex. /data/bert-base-summaries.json

    {
        "<document-id>": {
            "summary": "this is awesome",
            "metadata": {
                "tokens_ids": [1, 2, 3]",
                "token_entropy": [0.5, 0.3, 0.2]
            }
        }
    }

    Args:
        dataset: dataset name, i.e. "xsum"
        model: unique name of the model/paper that was used to generate the summary (in huggingface model.config.name_or_path)
        model_config:
            dictionary of the config that was used to generate the summary (in huggingface serialize using model.config.to_dict())
            if the model config is not available, but it was generated by a known paper this dict can reference the paper
        generated_summaries: dictionary of document id -> summary
        metadata: optional dict containing metadata for the generated summaries, for example token ids & entropy

    """
    store_model_config(dataset, model, model_config)
    dir = dataset_dir(dataset)
    path = model_path(dataset, model, "summaries.json")

    if not os.path.exists(path):
        os.makedirs(dir, exist_ok=True)
        stored_summaries = {}
    else:
        with open(path, "r") as f:
            stored_summaries = json.load(f)

    for document_id, summary in generated_summaries.items():
        stored_summaries[str(document_id)] = {
            "summary": summary,
            "metadata": metadata[document_id] if document_id in metadata else {},
        }

    with open(path, "w") as f:
        f.write(json.dumps(stored_summaries, indent=2))


def get_summaries(dataset: str, model: str):
    with open(f"{dataset_dir(dataset)}/{slugify(model)}-summaries.json", "r") as f:
        return json.load(f)
