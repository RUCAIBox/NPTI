# NPTI
This repository is the official implementation of our paper "Neuron-based Personality Trait Induction in Large Language Models".
## Dataset
To better identify personality-related neurons, we first constructed the PersonalityBench dataset, comprising 180,000 open-ended questions tailored to capture distinct personality traits based on Big Five personality theory. Specifically, we utilize the description from IPIP-NEO-300 questionnaire and common real-world topics introduced in UltraChat to generate the situational questions in PersonalityBench. The dataset is shown in `NPTI/datset`. 
## Personality-related Neurons Found by NPTI
Within a given layer, the FFN module can be expressed as:

$$**h** = \left( \sigma \left( \hat{**h**} **W**_1 \right) \odot \left(\hat{**h**} **W**_3 \right) \right)\cdot **W**_2,$$

where $\hat{**h**}$ in $\mathbb{R}^{d}$ represents the output of the MHA module for a specific token in this layer. The function $\sigma(\cdot)$ typically denotes a non-linear activation function, such as SiLU. The learned projection matrices are $**W**_1$ $\in$ $\mathbb{R}^{d \times d'}$, $**W**_2$ in $\mathbb{R}^{d' \times d}$, and $**W**_3$ \in \mathbb{R}^{d \times d'}$. In this context, a \emph{neuron} is conceptualized as applying a linear transformation to a specific column of the weight matrix $**W**_1$ followed by a non-linear activation function to the result.
We found the neurons that related to the positive/negative aspect of each personality trait. The neurons can be found in `NPTI/neuron_results`.
## Identifying Language-specific Neurons
To find personality-related neurons in LLaMA-8B-Instruct, you can excute:
```bash
bash NPTI/code/search_neuron.sh
```
## Using NPTI to modify the personality of LLM
To modify certain personality trait of LLM using NPTI, you can excute:  
```bash
bash NPTI/code/answer_question_change_neuron.sh
```
## Using ChatGPT to score the results:
To use ChatGPT to automatically assess the degree of expression of a specific personality trait and the fluency of each response, you can excute: 
```bash
python NPTI/code/gpt4_score.py
```
