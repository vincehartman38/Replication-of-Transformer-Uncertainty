# Replication of Transformer Uncertainty
This project replicates the results in the paper "[Understanding Neural Abstractive Summarization Models via Uncertainty](https://arxiv.org/abs/2010.07882)"
by Jiacheng Xu, Shrey Desai, and Greg Durrett.

There original code can be found at: [https://github.com/jiacheng-xu/text-sum-uncertainty](https://github.com/jiacheng-xu/text-sum-uncertainty)

## Setup
Setup (python 3.9.1). Clone the repository and install requirements.
```
git clone
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Datasets
Datasets are loaded from HuggingFace with the following commands:
```
from datasets import load_dataset

load_dataset("xsum")
load_dataset("cnn_dailymail")
```
The authors use "10K generation steps from PEGASUSCNN/DM, PEGASUSXSUM, BARTCNN/DM and BARTXSUM
respectively." Based on the graphs and the count, this implies that the authors stop generation
of the model after 10k output tokens in the summary sequence. The authors did not state if they
used the train, validation, or test dataset for their paper; I assumed the test dataset. Further,
the authors did not specify what is the subset of the test dataset; I assume the start at the
beginning of the test dataset and stop after the reach 10k generation steps.

## Models
Experiments use the two models PEGASUS and BART. I use HuggingFace for building these two models.
1. [PEGASUS CNN Dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
2. [PEGASUS XSum](https://huggingface.co/google/pegasus-xsum)
3. [BART Large CNN](https://huggingface.co/facebook/bart-large-cnn)
4. [BART Large XSum](https://huggingface.co/facebook/bart-large-xsum)

## Entropy
Entropy is calculated using the Shannon entropy for a probability distribution. Nucleus sampling
is used to sample only the top 1 - p most probable outcomes. Nucleus sampling is set through
a HuggingFace parameter during model generation. For example:
```
model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_scores=True,
        top_p=0.95,
    )
```
top_p is nucleus sampling in HuggingFace.

## Analysis
### Model Uncertainty during Generation
In this part, I attempt replication of the behavior of BART and PEGASUS in how they perform a mixture
of copying and generating tokens.

For the first figure, I separate the output bigrams into two buckets: (1) if they have been generated
from existing bigrams in the source document or (2) if they are novel bigrams generated by the model.

For this replication, I calculate the position of the output token with respect to a summary sentence.
Note that there might be multiple sentences for a summary. I create 10 buckets from 0.0 to 0.9 to
indicate what part of the sentence the token is located (where 0.0 is within the first 10% of the source
and 0.9 is the last 10% of the source). Figure 2 is calculated from the same source documents as Figure 1.

#### Figure 1 from Original Paper
![Original Bigram Prediction Engropy](https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/original_figures/replication_figure1.jpg)

#### Replicated Figures for Bigram Entropy Extracted/Novel

These results align with the original paper in that existing bigrams have lower entropy actions
than novel bigrams. In the figures, median entropy for Existing Bigrams is consistently closer to 0 
when compared to the median of Novel Bigrams (the Existing Bigram median is always to the left of the
Novel Bigram median). The XSum dataset for both PEGASUS and BART have more more novel bigrams than the 
CNN / DailyMail dataset; these models perform more copying. The XSum datasets erform more generation.
For XSum, the distributions tend to be closer to a normal distribuition than the CNN / DailyMail
distributions.

Images are listed in order of left to right compared to original one.  

DATASET | PEGASUS | BART 
:-------------------------:|:-------------------------:| :-------------------------:
CNN/DM | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-cnn_dailymail_hisotgram.jpeg" width=500 alt="Replication Figure 1 PEGASUS CNN"> | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-cnn_hisotgram.jpeg" width=500 alt="Replication Figure 1 BART CNN">
XSum | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-xsum_hisotgram.jpeg" width=500 alt="Replication Figure 1 PEGASUS XSUM">  |  <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-xsum_hisotgram.jpeg" width=500 alt="Replication Figure 1 BART XSUM">

#### Figure 2 from Original Paper
![Original Bigram Prediction Entropy](https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/original_figures/replication_figure2.jpg)

#### Replicated Figures for Entropy for Sentence Position
Simiar to the original paper, the entropy is higher at the beginning of the sentence and decreases
as the sentence progresses. Likewise, the Pegasus CNN model has very little entropy dispersion at the
0.9 bucket, which exactly matched the authors original figure. The 0.1 bucket is very distinctly above
the other buckets, also similar to the original figures.

DATASET | PEGASUS | BART 
:-------------------------:|:-------------------------:| :-------------------------:
CNN/DM | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-cnn_dailymail_boxplot.jpeg" width=500 alt="Replication Figure 2 PEGASUS CNN"> |<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-cnn_boxplot.jpeg" width=500 alt="Replication Figure 2 BART CNN">
XSum | <img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-xsum_boxplot.jpeg" width=500 alt="Replication Figure 2 PEGASUS XSUM"> |<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/bart-large-xsum_boxplot.jpeg" width=500 alt="Replication Figure 2 BART XSUM">

### Entropies of Syntactic Productions
I used the summaries generated from the first part with the [Berkely Neural Parser](https://github.com/nikitakit/self-attentive-parser)
(Benepar) and explore the connection between syntax and uncertainty. When supplied a summary,
the parser will create a linearized tree of the output

For example, here is the parser and linearized tree ouput for an XSum summary.

Summary: "Olympic long jump champion Greg Rutherford has qualified for Saturday's final at the World Championships in London."

Benepar: (S (NP (NML (NML (JJ Olympic) (JJ long) (NN jump)) (NN champion)) (NNP Greg) (NNP Rutherford)) (VP (VBZ has) (VP (VBN qualified) (PP (IN for) (NP (NP (NP (NNP Saturday) (POS 's)) (NN final)) (PP (PP (IN at) (NP (DT the) (NNP World) (NNPS Championships))) (PP (IN in) (NP (NNP London)))))))) (. .))

Linearized Tree: (((( Olympic  long  jump ) champion ) Greg  Rutherford )( has ( qualified ( for ((( Saturday ('s)) final )(( at ( the  World  Championships ))( in ( London )))))))(..))

Syntactic Distances would then be the following for this example:  
D(Rutherford, has) = len(")(") = 2  
D(Olympic, long) = len("") = 0  
D(London, .) = (len(")))))))(") = 8 -> placed in 5+ bucket  

#### Figure 3 from Original Paper
![Original Syntactic Distance](https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/original_figures/replication_figure3.jpg)

#### Replicated Figures for Syntactic Distance
I was not able to replicate these results. My results did not find a relationship between the
syntactic distance and entropy. I have verified the underlying code and the calculation
for syntactic distance and entropy; all calculations are correct. There maybe a few reasons why I was
unable to replicate these results:
* My subset of the XSum and CNN/DM datasets might not match the authors.
* Since the formula for the linearized tree was not provided; my implementation may not match the authors.
I emailed the head author, Jiacheng Xu, and he responded that Shrey Desai worked on this portion and he
will be gettin back to me.

CNN/DM | XSum 
:-------------------------:| :-------------------------:
<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-cnn_dailymail_syntactic.jpeg" width=500 alt="Replication Figure 3 PEGASUS XSUM"> |<img src="https://raw.githubusercontent.com/vincehartman38/Replication-of-Transformer-Uncertainty/main/results/pegasus-xsum_syntactic.jpeg" width=500 alt="Replication Figure 3 Pegasus CNN">
