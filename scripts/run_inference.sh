#!/bin/bash
GPU_IDS=3

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 inference.py \
    --model_path="model_outputs/lightning-template-epoch=02-val_loss=0.0842-v2.ckpt" \
    --config_path="config/dense_model.json" \
    --seed=42 \
    --accelerator=gpu \
    --devices=1 \
    --auto_select_gpus=true \
    --model_select=rnn \
    --truncated_bptt_steps=2