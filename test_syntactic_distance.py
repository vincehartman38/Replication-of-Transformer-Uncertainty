import benepar
import spacy
import ssl
import regex as re

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_trf")
    # benepar.download("benepar_en3")

    if spacy.__version__.startswith("2"):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    sentence = "Olympic long jump champion Greg Rutherford has qualified for Saturday's final at the World Championships in London."

    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    parser = sent._.parse_string
    print(parser)
    m = re.sub(r"\b[A-Z]+\b", "", parser)
    m = m.replace(" ", "")
    m = re.sub(r"\((\w+)\)", r" \1 ", m)
    print(m)
