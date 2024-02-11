#!/usr/bin/bash
accelerate launch \
    --config_file accelerate_one_gpu.yaml \
    train_main.py \
    --train_type sft \
    --train_files ./data/sft_train_data.parquet \
    --output_dir ./model_save/sft \
    --tokenizer_dir ./model_save/tokenizer \
    --train_from_model_dir ./model_save/pre/checkpoint-10 \
    --per_device_train_batch_size 6 \
    --num_train_epochs 8 \
    --learning_rate 1e-4
