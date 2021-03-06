# Introduction: Replication of Transformer Uncertainty
This project replicates the results in the paper "[Understanding Neural Abstractive Summarization Models via Uncertainty](https://arxiv.org/abs/2010.07882)" by Jiacheng Xu, Shrey Desai, and Greg Durrett.

There original code can be found at: [https://github.com/jiacheng-xu/text-sum-uncertainty](https://github.com/jiacheng-xu/text-sum-uncertainty)

Their slide deck is here: [https://github.com/jiacheng-xu/text-sum-uncertainty/blob/master/slide.pdf](https://github.com/jiacheng-xu/text-sum-uncertainty/blob/master/slide.pdf)

The author's original paper explores the relationship of entropy, or uncertainty, from summarization encoder-decoder models with different corresponding metrics: if the output tokens are novel or extracted from the source text, the sentence position of the output source token, and the relationship with the attention entropy. Through understanding how transformers perform generation and lead to the predicted entropy, we can analyze and design approaches to inform users about the nature of that uncertainty to create a more explainable system.

This replication paper is broken into an analysis of four result section graphs:
1. Output token entropy compared to if the token originated from the source text or not
2. Output token entropy compared to the sentence position
3. Output token entropy change compared to syntactic distance between tokens
4. Output token entropy compared to the underlying attention entropy

## Setup
Setup (python 3.9.1). Clone the repository and install requirements.
```
git clone https://github.com/vincehartman38/Replication-of-Transformer-Uncertainty.git
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## How to Run
First create the JSON data files for the four model which contain the summaries, tokens, and entropy level data. After generating all four JSON files, run the python scripts for generating the figures in the order listed.

Run scripts in this order to reproduce my results:
1. ```python generate_summary_data.py --model bart-large-xsum```
2. ```python generate_summary_data.py --model bart-large-cnn```
3. ```python generate_summary_data.py --model pegasus-xsum```
4. ```python generate_summary_data.py --model pegasus-cnn_dailymail```
5. ```python entropy.py --model bart-large-xsum```
6. ```python entropy.py --model bart-large-cnn```
7. ```python entropy.py --model pegasus-xsum```
8. ```python entropy.py --model pegasus-cnn_dailymail```
9. ```python syntactic_distance.py --model pegasus-xsum```
10. ```python syntactic_distance.py --model pegasus-cnn_dailymail```
11. ```python cross_attention.py```

# Method
To replicate this paper, I (1) collected the datasets from Huggingface, (2) loaded the four models from HuggingFace, (3) generated the underlying entropy metadata necessary for all the graphing results, and (4) designed the graphs to match as exactly as possible in Matplotlib based on the published paper.

## Datasets
Datasets are loaded from HuggingFace with the following commands:
```
from datasets import load_dataset

load_dataset("xsum")
load_dataset("cnn_dailymail")
```
The authors use "10K generation steps from PEGASUSCNN/DM, PEGASUSXSUM, BARTCNN/DM and BARTXSUM respectively." Based on the graphs and the count, this implies that the authors stop generation of the model after 10k output tokens in the summary sequence. The authors did not state if they used the train, validation, or test dataset for their paper; I assumed the test dataset. Further, the authors did not specify what is the subset of the test dataset; I assume the start at the beginning of the test dataset and stop after the reach 10k generation steps.

## Models
Experiments use the two models PEGASUS and BART. I use HuggingFace for building these two models trained with two datasets (4 total):
1. [PEGASUS CNN Dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
2. [PEGASUS XSum](https://huggingface.co/google/pegasus-xsum)
3. [BART Large CNN](https://huggingface.co/facebook/bart-large-cnn)
4. [BART Large XSum](https://huggingface.co/facebook/bart-large-xsum)

## Entropy
Entropy is calculated using the Shannon entropy for a probability distribution. Nucleus sampling is used to sample only the top 1 - p most probable outcomes. Nucleus sampling is set through a HuggingFace parameter during model generation. To compute the mean encoder attention entropy compared to a single output prediction entropy, I use the cross-attentions from the model outputs which is also set in HuggingFace
```
model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_scores=True,
        **top_p=0.95,**
        **output_attentions=True,**
    )
```
output_scores:
top_p: nucleus sampling in HuggingFace.  
output_attentions: returns attentions of all layers

# Results
### Model Uncertainty during Generation
In this part, I attempt replication of the behavior of BART and PEGASUS in how they perform a mixture of copying and generating tokens in Figures 1 and 2.

For the first figure, I separate the output bigrams into two buckets: (1) if they have been generated from existing bigrams in the source document or (2) if they are novel bigrams generated by the model.

For this replication, I calculate the position of the output token with respect to a summary sentence. Note that there might be multiple sentences for a summary. I create 10 buckets from 0.0 to 0.9 to indicate what part of the sentence the token is located (where 0.0 is within the first 10% of the source and 0.9 is the last 10% of the source). Figure 2 is calculated from the same source documents as Figure 1.

#### Figure 1 from Original Paper
<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/original_figures/replication_figure1.jpg" width=500 alt="Original Bigram Prediction Entropy">

#### Replicated Figures for Bigram Entropy Extracted/Novel
DATASET | PEGASUS | BART 
:-------------------------:|:-------------------------:| :-------------------------:
CNN/DM | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-cnn_dailymail_hisotgram.jpeg" width=250 alt="Replication Figure 1 PEGASUS CNN"> | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-cnn_hisotgram.jpeg" width=250 alt="Replication Figure 1 BART CNN">
XSum | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-xsum_hisotgram.jpeg" width=250 alt="Replication Figure 1 PEGASUS XSUM">  |  <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-xsum_hisotgram.jpeg" width=250 alt="Replication Figure 1 BART XSUM">

These results align with the original paper in that existing bigrams have lower entropy actions than novel bigrams. In the figures, median entropy for Existing Bigrams is consistently closer to 0  when compared to the median of Novel Bigrams (the Existing Bigram median is always to the left of the Novel Bigram median). The XSum dataset for both PEGASUS and BART have more more novel bigrams than the  CNN / DailyMail dataset; these models perform more copying. The XSum datasets perform more generation. For XSum, the distributions tend to be closer to a normal distribution than the CNN / DailyMail distributions.

#### Figure 2 from Original Paper
<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/original_figures/replication_figure2.jpg" width=500 alt="Original Sentence Position Entropy">

#### Replicated Figures for Entropy for Sentence Position
DATASET | PEGASUS | BART 
:-------------------------:|:-------------------------:| :-------------------------:
CNN/DM | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-cnn_dailymail_boxplot.jpeg" width=250 alt="Replication Figure 2 PEGASUS CNN"> |<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-cnn_boxplot.jpeg" width=250 alt="Replication Figure 2 BART CNN">
XSum | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-xsum_boxplot.jpeg" width=250 alt="Replication Figure 2 PEGASUS XSUM"> |<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-xsum_boxplot.jpeg" width=250 alt="Replication Figure 2 BART XSUM">

Similar to the original paper, the entropy is higher at the beginning of the sentence and decreases as the sentence progresses. Likewise, the Pegasus CNN model has very little entropy dispersion at the 0.9 bucket, which exactly matched the authors original figure. The 0.1 bucket is very distinctly above the other buckets, also similar to the original figures.

### Entropies of Syntactic Productions
I used the summaries generated from the first part with the [Berkely Neural Parser](https://github.com/nikitakit/self-attentive-parser) (Benepar) and explore the connection between syntax and uncertainty. The authors graph this correlation this in a vertical box-plot in Figure 3. The creation of the Linearized Tree from Benepar is created from referencing the authors PDF slides (not specifically stated in their paper).

For example, here is the parser and linearized tree output for an XSum summary.

Summary: "Olympic long jump champion Greg Rutherford has qualified for Saturday's final at the World Championships in London."

Benepar: (S (NP (NML (NML (JJ Olympic) (JJ long) (NN jump)) (NN champion)) (NNP Greg) (NNP Rutherford)) (VP (VBZ has) (VP (VBN qualified) (PP (IN for) (NP (NP (NP (NNP Saturday) (POS 's)) (NN final)) (PP (PP (IN at) (NP (DT the) (NNP World) (NNPS Championships))) (PP (IN in) (NP (NNP London)))))))) (. .))

Linearized Tree: (((( Olympic  long  jump ) champion ) Greg  Rutherford )( has ( qualified ( for ((( Saturday ('s)) final )(( at ( the  World  Championships ))( in ( London )))))))(..))

Syntactic Distances would then be the following for this example according to the author's definition in their paper:  
D(Rutherford, has) = len(")(") = 2  
D(Olympic, long) = len("") = 0  
D(London, .) = (len(")))))))(") = 8 -> placed in 5+ bucket  

#### Figure 3 from Original Paper
<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/original_figures/replication_figure3.jpg" width=500 alt="Original Syntactic Distance">

#### Replicated Figures for Syntactic Distance
CNN/DM | XSum 
:-------------------------:| :-------------------------:
<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-cnn_dailymail_syntactic.jpeg" width=300 alt="Replication Figure 3 PEGASUS XSUM"> |<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-xsum_syntactic.jpeg" width=300 alt="Replication Figure 3 Pegasus CNN">

I was not able to replicate these results as my results did not find a relationship between the syntactic distance and entropy. I have verified the underlying code and the calculation for syntactic distance and entropy; all calculations are correct. There maybe a few reasons why I was unable to replicate these results:
* My subset of the XSum and CNN/DM datasets might not match the authors. The original results may have just been noise, and when a slightly new subset of the XSum and CNN/DM datasets are used, there is in fact no replicated relationship between sentence syntax and entropy change.
* Since the formula for the linearized tree was not provided; my implementation may not match the authors.
I emailed the head author, Jiacheng Xu, and he responded that Shrey Desai worked on this portion and he will be getting back to me.

### Attention Entropy
The authors explored if there is a relationship between the entropy in the encoder and the prediction entropy from the decoder. The goal is to see how an encoder places attention during generation and if it correlates with prediction.

They computed the mean value of the attention entropy within each bucket of prediction entropy in Figure 4.

#### Figure 4 from Original Paper
<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/original_figures/replication_figure4.jpg" width=500 alt="Original Attention Entropy">

#### Replicated Figure for Attention Entropy and Prediction Entropy
<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/attention_and_entropy.jpeg" width=500 alt="Replication Figure 3 Pegasus CNN">
For my replication, my results are comparable to the authors. I use the cross attention matrix and find the mean across all layers and heads for each token in the output summary. Similar to the authors, when the prediction entropy is around 2, the attention entropy has saturated except for the BART CNN/DM model. For my replication, I did not implement their tf-idf proposed method to discard the 5% of the tokens from the source ocument with the attention values of tokens with the highest f score. The authors proposed method for reducing the low-information tokens does not appear to have been impactful.
