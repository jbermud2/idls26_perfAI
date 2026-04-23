#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Error: WANDB_API_KEY is not set."
  echo "Set it first, e.g. export WANDB_API_KEY='your_key_here'"
  exit 1
fi

python "${SCRIPT_DIR}/main.py" \
  --data-root /ocean/projects/cis260045p/shared/data \
  --use-wandb \
  --wandb-api-key "${WANDB_API_KEY}" \
  --dendrite-mode 1 \
  --max-dendrites 4 \
  --pai-forward-function relu \
  --improvement-threshold 1 \
  --candidate-weight-init-mult 0.1 \
  --model efficientnet_b4 \
  > "${OUTPUT_FILE}" 2>&1

echo "Run completed. Logs: ${OUTPUT_FILE}"
