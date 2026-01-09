#! /usr/bin/env bash

# Data Paths
DATASET_PATH=data/MSC
SLM_TRIPLET_PATH=${DATASET_PATH}/qwen_msc_triplet.json

# API Key Setup (Please enter your key)
API_KEY=

# Run Evaluation
python -u main.py \
    --dataset msc \
    --data_path ${DATASET_PATH} \
    --data_name sequential_test.json \
    --api_key "${API_KEY}" \
    --model gpt-4o-mini \
    --usr_name SPEAKER_1 --agent_name SPEAKER_2 \
    --test_num 1 \
    --relevance_memory_number 3 \
    --triplet_mode slm \
    --slm_data_path ${SLM_TRIPLET_PATH} \
    --log_name msc_eval_result.txt