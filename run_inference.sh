#!/bin/bash
GPU_IDS=3

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 ./inference.py \
    --model_path="../models/" \
    --seed=42 \
    --accelerator=gpu \
    --devices=1 \
    --auto_select_gpus=true \
    --input_dense_dim=512 \
    --output_dense_dim=256