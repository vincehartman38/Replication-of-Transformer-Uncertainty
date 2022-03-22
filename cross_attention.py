import math
import numpy as np
from storage import get_summaries
from utils import load_model_and_tokenizer
from figures import create_attention_graph


if __name__ == "__main__":
    model_names = [
        "pegasus-cnn_dailymail",
        "pegasus-xsum",
        "bart-large-cnn",
        "bart-large-xsum",
    ]
    prediction_entropies = {key: {} for key in model_names}
    for m_name in model_names:
        model, tokenizer = load_model_and_tokenizer(m_name)
        summaries = get_summaries(m_name, model.config.name_or_path)

        prediction_entropies[m_name] = {
            key: [] for key in np.round(np.linspace(0.0, 4.5, 10), 1)
        }

        for key, value in summaries.items():
            if "attention_entropy" in value["metadata"]:
                entropies = value["metadata"]["entropy"]
                attention_entropies = value["metadata"]["attention_entropy"]
                for ind, e in enumerate(entropies):
                    ae = attention_entropies[ind]
                    rounded_e = math.floor(e * 2) / 2.0
                    rounded_e = 4.5 if rounded_e >= 4.5 else rounded_e
                    prediction_entropies[m_name][rounded_e].append(ae)

    create_attention_graph(prediction_entropies)
