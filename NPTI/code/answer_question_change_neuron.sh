#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="/home/dengjia/model_weights/llama3-8b-inst"
NEURON_DIR="NPTI/neuron_results"
OUTPUT_DIR="NPTI/answer_results_cn"
QUESTION_DIR="NPTI/dataset/test"
BATCH_SIZE=32
PYTHON_FILE="NPTI/code/answer_question_change_neuron.py"  # 指定你的 Python 文件名

# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --neuron_dir $NEURON_DIR \
  --output_dir $OUTPUT_DIR \
  --question_dir $QUESTION_DIR \
  --batch_size $BATCH_SIZE
