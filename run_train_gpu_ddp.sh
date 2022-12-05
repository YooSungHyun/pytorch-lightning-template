#!/bin/bash
GPU_IDS=3

OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --standalone --nnodes=1 --nproc_per_node=1 ./train.py \
    --output_dir="../models/" \
    --data_dir="" \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --val_check_interval=0.25 \
    --accumulate_grad_batches=1 \
    --max_epochs=25 \
    --log_every_n_steps=1 \
    --accelerator=gpu \
    --strategy=ddp \
    --num_nodes=1 \
    --replace_sampler_ddp=false \
    --devices=1 \
    --auto_select_gpus=true \
    --auto_scale_batch_size=false \
    --learning_rate=0.00005 \
    --max_lr=0.0001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.2 \
    --ratio=0.2 \
    --div_factor=10 \
    --final_div_factor=10 \
    --input_dense_dim=512 \
    --output_dense_dim=256 \
    --valid_on_cpu=true \
    --model_select=rnn \
    --truncated_bptt_steps=1
