# NPTI
This repository is the official implementation of our paper "Neuron-based Personality Trait Induction in Large Language Models".
## Dataset
To better identify personality-related neurons, we firstly constructed the \textsc{PersonalityBench} dataset, comprising 180,000 open-ended questions tailored to capture distinct personality traits based on Big Five personality theory. Specifically, we utilize the \textsc{IPIP-NEO-300} questionnaire~\citep{goldberg1999broad,goldberg2006international} to generate the situational questions in \textsc{PersonalityBench}. 
## Personality-related Neurons Found by NPTI
In this work , we propose the novel NPTI method, which can effectively perform personality trait induction for LLMs. The dataset is shown in `NPTI/datset`. Based on \textsc{PersonalityBench}, we designed an efficient identification method for personality-related neurons.
## Identifying Language-specific Neurons
To find personality-related neurons in LLaMA-8B-Instruct,you can excute:
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
