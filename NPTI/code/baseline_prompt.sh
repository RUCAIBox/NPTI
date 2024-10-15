#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="model_weights/llama3-8b-inst"
PERSONALITY_DESCRIPTION="NPTI/dataset/description.json"
OUTPUT_FOLDER="NPTI/answer_results_cp"
QUESTION_DIR="NPTI/dataset/test"
BATCH_SIZE=32

# Run the Python script with the specified parameters
python NPTI/code/baseline_prompt.py \
    --model "$MODEL_PATH" \
    --personality_description "$PERSONALITY_DESCRIPTION" \
    --output_folder "$OUTPUT_FOLDER" \
    --question_dir $QUESTION_DIR \
    --batch_size "$BATCH_SIZE"
