#!/bin/bash
# =============================================================================
# Template job SLURM per il training sulla VM universitaria (GPU T4).
# Uso:
#   sbatch scripts/slurm/train.sbatch configs/resnet50.yaml
# Personalizzare partizione/account secondo la configurazione del cluster.
# =============================================================================
#SBATCH --job-name=genimage-train
#SBATCH --output=experiments/slurm-%j.out      # %j = job id
#SBATCH --error=experiments/slurm-%j.err
#SBATCH --gres=gpu:1                            # 1x T4
#SBATCH --cpus-per-task=8                       # workers DataLoader
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

CONFIG="${1:?Uso: sbatch train.sbatch <config.yaml>}"
DATA_ROOT="${2:-data}"                          # default: ./data sulla VM

echo "Job $SLURM_JOB_ID su $(hostname)"
echo "Config: $CONFIG | data-root: $DATA_ROOT"

# --- Ambiente (adattare al cluster: module load / conda activate) ---
# module load cuda/12.1
source ~/grossi/.venv/bin/activate


python -c "import torch; print('CUDA disponibile:', torch.cuda.is_available())"

# --- Training ---
python -m src.train --config "$CONFIG" --data-root "$DATA_ROOT"

echo "Job completato."
