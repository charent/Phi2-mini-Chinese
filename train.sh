#!/bin/bash
accelerate launch --multi_gpu --num_processes 2 --config_file accelerate_multi_gpu.yaml $1  
# accelerate launch $1

# 使用方式，后面接一个训练脚本即可， 如：./train.sh sft.py