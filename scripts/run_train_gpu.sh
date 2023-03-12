#!/bin/bash
GPU_IDS=0

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 ./train.py \
    --output_dir="../models/" \
    --data_dir="" \
    --config_path="config/dense_model.json" \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --val_check_interval=0.25 \
    --accumulate_grad_batches=1 \
    --max_epochs=3 \
    --log_every_n_steps=1 \
    --accelerator=gpu \
    --replace_sampler_ddp=false \
    --devices=1 \
    --auto_scale_batch_size=false \
    --learning_rate=0.00005 \
    --max_lr=0.0001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.2 \
    --ratio=0.2 \
    --div_factor=10 \
    --final_div_factor=10 \
    --valid_on_cpu=false \
    --model_select=linear \
    --dropout_p=0.1 \
    --cutoff_epoch=1 \
    --drop_mode=standard \
    --drop_schedule=constant
