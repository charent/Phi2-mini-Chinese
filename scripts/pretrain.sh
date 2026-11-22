#!/usr/bin/env bash
# 预训练：在仓库根目录执行 `bash scripts/pretrain.sh`
# 重构后不再 `cd ..`，请先 `uv sync --extra train`；额外参数可直接追加在后面。
set -euo pipefail

uv run accelerate launch \
    --config_file configs/accelerate_one_gpu.yaml \
    -m phi2_mini \
    --train_type pre \
    --train_files data/baike_no_duplicate.parquet \
    --output_dir model_save/pre \
    --tokenizer_dir model_save/tokenizer \
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
