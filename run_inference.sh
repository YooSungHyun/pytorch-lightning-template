#!/bin/bash
GPU_IDS=3

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 ./inference.py \
    --model_path="../models/lightning-template-epoch=07-val_loss=0.0840.ckpt" \
    --seed=42 \
    --accelerator=gpu \
    --devices=1 \
    --auto_select_gpus=true \
    --input_dense_dim=512 \
    --output_dense_dim=256 \
    --model_select=rnn \
    --truncated_bptt_steps=2