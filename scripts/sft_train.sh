#!/usr/bin/env bash
# SFT 微调：在仓库根目录执行 `bash scripts/sft_train.sh`
set -euo pipefail

uv run accelerate launch \
    --config_file configs/accelerate_one_gpu.yaml \
    -m phi2_mini \
    --train_type sft \
    --train_files data/sft_train_data.parquet \
    --output_dir model_save/sft \
    --tokenizer_dir model_save/tokenizer \
    --train_from_model_dir model_save/pre/checkpoint-10 \
    --bf16 true \
    --fp16 false \
    --eval_steps 2000 \
    --evaluation_strategy "epoch" \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-4 \
    --log_level info \
    --logging_first_step true \
    --logging_steps 5 \
    --num_train_epochs 4 \
    --optim "lion_32bit" \
    --report_to "tensorboard" \
    --save_steps 5000 \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    "$@"
