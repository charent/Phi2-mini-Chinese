#!/usr/bin/bash
cd ..
accelerate launch \
    --config_file accelerate_one_gpu.yaml \
    train_main.py \
    --train_type sft \
    --train_files ./data/sft_train_data.parquet \
    --output_dir ./model_save/sft \
    --tokenizer_dir ./model_save/tokenizer \
    --train_from_model_dir ./model_save/pre/checkpoint-10 \
    --bf16 true \
    --fp16 false \
    --eval_steps 2000 \
    --evaluation_strategy "epoch" \
    --gradient_accumulation_steps 32 \
    --group_by_length true \
    --learning_rat 5e-4 \
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
    --per_device_train_batch_size 4 