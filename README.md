# MASC 576 Lab 3: DFT and MLIP Workflow

This repository contains the scripts, processed data, and result summaries for MASC 576 Lab 3. The project combines density functional theory (DFT), DFT molecular dynamics, and machine-learned interatomic potential (MLIP) fine-tuning using MACE.

## Overview

The lab consists of two main parts:

1. Review and summarize core DFT concepts, including primitive cells, k-point sampling, plane-wave cutoff convergence, equation-of-state fitting, bulk modulus, and electronic band structure.

2. Generate a DFT-labeled dataset for a periodic carbon structure, fine-tune a MACE foundation model, and evaluate the resulting MLIP against GPAW reference calculations.

## Workflow

The computational workflow for Task 2 was:

1. Select a periodic carbon configuration.
2. Relax the structure using GPAW.
3. Run DFT molecular dynamics at 300 K, 600 K, and 900 K.
4. Save sampled configurations with DFT energies and forces in extended XYZ format.
5. Fine-tune a small MACE foundation model using the generated dataset.
6. Evaluate the fine-tuned model using parity plots, equation-of-state curves, and bulk modulus comparison.

## Repository Structure

```text
scripts/
  gpawrun.py              DFT relaxation and DFT-MD data generation
  gpaw_eos.py             GPAW equation-of-state calculation
  mace_eos.py             MACE equation-of-state calculation
  mace_parity.py          Parity plot evaluation
  sample_finetune.sh      MACE fine-tuning command
  slurm_finetune_gpu.sh   SLURM script for GPU fine-tuning

data/
  sample_config.xyz       Initial carbon configuration
  relax.xyz               Relaxed carbon structure
  training_data.extxyz    DFT-labeled training dataset

figures/
  gpaw_eos.png            GPAW equation-of-state plot
  mace_eos.png            MACE equation-of-state plot
  mace_parity.png         Energy and force parity plot
  gpaw_eos_summary.txt    GPAW EOS fit summary
  mace_eos_summary.txt    MACE EOS fit summary
  mace_parity_summary.txt Parity evaluation summary

results/
  maceft_sample75C_run-123.log  MACE fine-tuning log
