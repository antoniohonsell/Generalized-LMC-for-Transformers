#!/bin/bash
#SBATCH --job-name="tinyshakes_muon"
#SBATCH --account=3199937
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antonio.honsell@studbocconi.it

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"

module load miniconda3

eval "$(conda shell.bash hook)"
conda activate lmc

python --version
nvidia-smi

python train_muon.py \
  --seed 0 \
  --epochs 100 \
  --early_stop \
  --early_stop_patience 5 \
  --splits_dir splits_tiny_contig \
  --muon_lr 0.02 \
  --muon_momentum 0.95 \
  --lr 3e-4 \
  --wandb \
  --wandb_project gpt2-merging-demo \
  --wandb_group tinyshakes_muon \
  --wandb_tags muon,seed0,nembd256 \
  --n_embd 256

python train_muon.py \
  --seed 1 \
  --epochs 100 \
  --early_stop \
  --early_stop_patience 5 \
  --splits_dir splits_tiny_contig \
  --muon_lr 0.02 \
  --muon_momentum 0.95 \
  --lr 3e-4 \
  --wandb \
  --wandb_project gpt2-merging-demo \
  --wandb_group tinyshakes_muon \
  --wandb_tags muon,seed1,nembd256 \
  --n_embd 256

python train_muon.py \
  --seed 2 \
  --epochs 100 \
  --early_stop \
  --early_stop_patience 5 \
  --splits_dir splits_tiny_contig \
  --muon_lr 0.02 \
  --muon_momentum 0.95 \
  --lr 3e-4 \
  --wandb \
  --wandb_project gpt2-merging-demo \
  --wandb_group tinyshakes_muon \
  --wandb_tags muon,seed2,nembd512 \
  --n_embd 512

echo "Finished at: $(date)"
