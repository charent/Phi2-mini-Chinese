@echo off
REM SFT 微调：在仓库根目录执行 scripts\sft_train.bat
chcp 65001 >nul
echo [phi2-mini] SFT 微调中...
uv run accelerate launch ^
    --config_file configs/accelerate_one_gpu.yaml ^
    -m phi2_mini ^
    --train_type sft ^
    --train_files data/sft_train_data.parquet ^
    --output_dir model_save/sft ^
    --tokenizer_dir model_save/tokenizer ^
    --train_from_model_dir model_save/pre/checkpoint-10 ^
    --per_device_train_batch_size 6 ^
    --num_train_epochs 8 ^
    --learning_rate 1e-4
