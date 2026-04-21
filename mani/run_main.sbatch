#!/bin/bash
#SBATCH --job-name=perfAI_effb5_pets
#SBATCH --account=cis260045p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=========================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Job name:       $SLURM_JOB_NAME"
echo "Node:           $(hostname)"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time:     $(date)"
echo "=========================================="

cd /ocean/projects/cis260045p/jbermude/idls26_perfAI/mani

mkdir -p logs

module purge
module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate /ocean/projects/cis260045p/shared/perf_all

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Perforated Backpropagation: install `perforatedbp`, then set (e.g. from a private file, not in git):
#   export PAIEMAIL='...'
#   export PAITOKEN='...'
# and use --dendrite-mode 2 below instead of 1.

DATA_ROOT="/ocean/projects/cis260045p/shared/data"

# Add --use-wandb (and export WANDB_API_KEY or pass --wandb-api-key) to log metrics to W&B.
python main.py \
    --data-root "$DATA_ROOT" \
    --dendrite-mode 1 \
    --max-dendrites 5 \
    --pai-forward-function relu \
    --improvement-threshold 0.5 \
    --candidate-weight-init-mult 0.1 \
    --epochs 100 \
    "$@"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
