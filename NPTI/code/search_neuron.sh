#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# Define parameters
MODEL_PATH="model_weights/llama3-8b-inst"
QUESTION_DIR="NPTI/dataset/search"
ANSWER_DIR="NPTI/answer_results"
NEURON_DIR="NPTI/neuron_results"
PERSONALITY_DESCRIPTION="NPTI/dataset/description.json"
BATCH_SIZE=32

# Run the Python script with arguments
python NPTI/code/search_neuron.py \
    --model "$MODEL_PATH" \
    --question_dir "$QUESTION_DIR" \
    --answer_dir "$ANSWER_DIR" \
    --neuron_dir "$NEURON_DIR" \
    --personality_desc "$PERSONALITY_DESCRIPTION" \
    --batch_size $BATCH_SIZE
python NPTI/code/process_neuron.py\
    --neuron_dir "$NEURON_DIR" \
