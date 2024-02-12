@echo off
cd ..
accelerate launch ^
    --config_file accelerate_one_gpu.yaml ^
    train_main.py ^
    --train_type pre ^
    --train_files ./data/baike_no_duplicate.parquet ./data/wiki_25w.parquet^
    --output_dir ./model_save/pre ^
    --tokenizer_dir ./model_save/tokenizer
