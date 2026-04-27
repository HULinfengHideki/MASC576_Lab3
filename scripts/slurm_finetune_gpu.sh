#!/bin/bash
#SBATCH -J maceft75C
#SBATCH -p warshel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=b17-11
#SBATCH -t 24:00:00
#SBATCH -A warshel_155
#SBATCH -o slurm-maceft-%j.out
#SBATCH -e slurm-maceft-%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

workdir=/project2/knomura_125/26S/MASC576/harry/lab3/Scripts_lab3

cd "$workdir"
source "${workdir}/.venv_mace/bin/activate"

echo "=========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "TMPDIR = ${TMPDIR:-not_set}"
echo "=========================================="

echo "Job started on $(date)"
echo "Running in: $(pwd)"
echo "HOST=$(hostname)"
echo "Python: $(which python)"
python --version
echo "mace_run_train: $(which mace_run_train)"

stdbuf -oL -eL mace_run_train \
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

echo "Job finished on $(date)"