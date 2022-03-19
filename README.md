# Replication of Transformer Uncertainty
This project replicates the results in the paper "[Understanding Neural Abstractive Summarization Models via Uncertainty](https://arxiv.org/abs/2010.07882)"
by Jiacheng Xu, Shrey Desai, and Greg Durrett.

There original code can be found at: [https://github.com/jiacheng-xu/text-sum-uncertainty](https://github.com/jiacheng-xu/text-sum-uncertainty)

## Setup
Setup (python 3.9.1). Clone the repository and install requirements.
```
git clone
pip install -r requirements.txt
```

## Datasets
Datasets are loaded from HuggingFace with the following commands:
```
from datasets import load_dataset

load_dataset("xsum")
load_dataset("cnn_dailymail")
```

## Models
Experiments use the two models PEGASUS and BART. I use HuggingFace for building these two models.
1. [PEGASUS CNN Dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
2. [PEGASUS XSum](https://huggingface.co/google/pegasus-xsum)
3. [BART Large CNN](https://huggingface.co/facebook/bart-large-cnn)
4. [BART Large XSum](https://huggingface.co/facebook/bart-large-xsum)

## Entropy
