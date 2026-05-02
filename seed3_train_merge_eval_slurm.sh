#!/bin/bash
#SBATCH --job-name="tinyshakes_seed3_full"
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

# ── Config ───────────────────────────────────────────────────────────────────
# Train a third seed (seed=3) with both AdamW and Muon at n_embd=256, then run
# orthogonal LMC merges pairing the new models with seeds 0 and 1 from both
# optimizer families, plus the same-seed cross merge (a3 × m3).
#
# All merges are same-width (256), orthogonal alignment.  W&B groups match the
# existing ones so the notebook plots pick up the new runs automatically.

SEED=3
NEMBD=256

A0="./gpt2_tinyshakespeare_seed0_nembd${NEMBD}"
A1="./gpt2_tinyshakespeare_seed1_nembd${NEMBD}"
A3="./gpt2_tinyshakespeare_seed${SEED}_nembd${NEMBD}"
M0="./gpt2_muon_tinyshakespeare_seed0_nembd${NEMBD}"
M1="./gpt2_muon_tinyshakespeare_seed1_nembd${NEMBD}"
M3="./gpt2_muon_tinyshakespeare_seed${SEED}_nembd${NEMBD}"

# ── Helpers ──────────────────────────────────────────────────────────────────

run_merge() {
  # $1 model_dir_0  $2 model_dir_1  $3 output_dir  $4 wandb_group
  echo ""
  echo "▶ Merging  $1  ×  $2  →  $3"
  python train_merger.py \
    --model_dir_0 "$1" \
    --model_dir_1 "$2" \
    --tokenizer_dir ./gpt2_tokenizer \
    --output_dir "$3" \
    --epochs 10 \
    --batch_size 32 \
    --fp16 \
    --splits_dir splits_tiny_contig \
    --early_stop \
    --sampler NARROW_UNIFORM \
    --wandb \
    --wandb_project gpt2-merging-demo \
    --wandb_group "$4"
}

run_eval() {
  # $1 merged_dir  $2 wandb_group
  echo ""
  echo "▶ Evaluating  $1"
  python eval.py \
    --merged_model_dir "$1" \
    --output_dir "eval_${1}" \
    --wandb \
    --wandb_project gpt2-merging-demo \
    --wandb_group "$2"
}

# ── 1) Train the two new seed-3 models ───────────────────────────────────────

echo ""
echo "▶ Training AdamW seed=${SEED} (n_embd=${NEMBD})"
python train.py \
  --seed ${SEED} \
  --epochs 100 \
  --early_stop \
  --early_stop_patience 5 \
  --splits_dir splits_tiny_contig \
  --wandb \
  --wandb_project gpt2-merging-demo \
  --wandb_group tinyshakes \
  --n_embd ${NEMBD}

echo ""
echo "▶ Training Muon seed=${SEED} (n_embd=${NEMBD})"
python train_muon.py \
  --seed ${SEED} \
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
  --wandb_tags muon,seed${SEED},nembd${NEMBD} \
  --n_embd ${NEMBD}

# ── 2) Within-AdamW merges: a3 × a0, a3 × a1 ─────────────────────────────────

run_merge "$A3" "$A0" merge_seed_3_0 tinyshakes-merge
run_merge "$A3" "$A1" merge_seed_3_1 tinyshakes-merge

# ── 3) Within-Muon merges: m3 × m0, m3 × m1 ──────────────────────────────────

run_merge "$M3" "$M0" merge_muon_seed_3_0 tinyshakes-muon-merge
run_merge "$M3" "$M1" merge_muon_seed_3_1 tinyshakes-muon-merge

# ── 4) Cross merges (AdamW × Muon, all 256-d) ────────────────────────────────
# New seed-3 paired against existing seeds 0 and 1, plus same-seed a3 × m3.

run_merge "$A3" "$M0" merge_cross_a3_m0 tinyshakes-cross-merge
run_merge "$A3" "$M1" merge_cross_a3_m1 tinyshakes-cross-merge
run_merge "$A0" "$M3" merge_cross_a0_m3 tinyshakes-cross-merge
run_merge "$A1" "$M3" merge_cross_a1_m3 tinyshakes-cross-merge
run_merge "$A3" "$M3" merge_cross_a3_m3 tinyshakes-cross-merge

# ── 5) Evaluate every new merge ──────────────────────────────────────────────

for d in merge_seed_3_0 merge_seed_3_1; do
  run_eval "$d" tinyshakes-eval
done

for d in merge_muon_seed_3_0 merge_muon_seed_3_1; do
  run_eval "$d" tinyshakes-muon-eval
done

for d in merge_cross_a3_m0 merge_cross_a3_m1 merge_cross_a0_m3 merge_cross_a1_m3 merge_cross_a3_m3; do
  run_eval "$d" tinyshakes-cross-eval
done

echo ""
echo "Finished at: $(date)"
