@echo off
REM 预训练：在仓库根目录执行 scripts\pretrain.bat（不再 cd ..）
chcp 65001 >nul
echo [phi2-mini] 预训练中...
uv run accelerate launch ^
    --config_file configs/accelerate_one_gpu.yaml ^
    -m phi2_mini ^
    --train_type pre ^
    --train_files data/baike_no_duplicate.parquet data/wiki_25w.parquet ^
    --output_dir model_save/pre ^
    --tokenizer_dir model_save/tokenizer
