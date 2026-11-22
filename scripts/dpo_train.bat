@echo off
REM DPO 偏好优化：在仓库根目录执行 scripts\dpo_train.bat
chcp 65001 >nul
echo [phi2-mini] DPO 优化中...
uv run accelerate launch ^
    --config_file configs/accelerate_one_gpu.yaml ^
    -m phi2_mini ^
    --train_type dpo ^
    --train_files data/dpo_train_data.json ^
    --output_dir model_save/dpo ^
    --tokenizer_dir model_save/tokenizer ^
    --train_from_model_dir model_save/sft/checkpoint-10 ^
    --per_device_train_batch_size 2 ^
    --learning_rate 1e-4
