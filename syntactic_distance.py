import benepar
import spacy
import regex as re
import ssl
import argparse
from storage import get_summaries
from utils import load_model_and_tokenizer, get_change
from figures import create_syntactic_boxplot

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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
        ],
        help="choose the model",
    )

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    summaries = get_summaries(args.model, model.config.name_or_path)

    nlp = spacy.load("en_core_web_trf")
    # benepar.download("benepar_en3")

    if spacy.__version__.startswith("2"):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    syn_distance = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    for key, value in summaries.items():
        doc = nlp(value["summary"][0])
        sents = list(doc.sents)
        syntactic_tree = []
        for s in sents:
            syntactic_tree.append(s._.parse_string)
        syntactic_tree = " ".join(syntactic_tree)
        t = value["metadata"]["tokens"]
        e = value["metadata"]["entropy"]
        for i in range(len(t) - 1):
            tot_parens = 0
            pairs = (t[i], t[i + 1])
            t1, t2 = t[i], t[i + 1]
            e_change = get_change(e[i], e[i + 1])
            t1_esc, t2_esc = re.escape(t1), re.escape(t2)
            m = re.search(t1_esc + "(.*?)" + t2_esc, syntactic_tree)
            new_start = syntactic_tree.find(t2)
            syntactic_tree = syntactic_tree[new_start:]
            if m:
                found = m.group(1)
                tot_parens = max(found.count("("), found.count(")"))
                tot_parens = 5 if tot_parens >= 5 else tot_parens
                syn_distance[tot_parens].append(e_change)
    create_syntactic_boxplot(syn_distance, args.model)
