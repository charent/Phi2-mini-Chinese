#!/usr/bin/bash
accelerate launch \
    --config_file accelerate_one_gpu.yaml \
    train_main.py \
    --train_type dpo \
    --train_files ./data/dpo_train_data.json \
    --output_dir ./model_save/dpo \
    --tokenizer_dir ./model_save/tokenizer \
    --train_from_model_dir ./model_save/sft/checkpoint-10 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4
