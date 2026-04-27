#!/bin/bash
set -euo pipefail

source .venv/bin/activate

mace_run_train \
    --name maceft_sample75C \
    --train_file output/sample_config.xyz/training_data.extxyz \
    --foundation_model small \
    --lr 0.001 \
    --valid_fraction 0.05 \
    --batch_size 10 \
    --energy_key energy \
    --forces_key forces \
    --E0s average \
    --multiheads_finetuning=False \
    --ema \
    --ema_decay=0.99 \
    --default_dtype=float64 \
    --device cuda \
    --max_num_epochs=100