#!/bin/bash

# PerforatedAI DistributedDataParallel Training Script
# Handles automatic restarting when dendrites are added (model restructured).

SAVE_NAME="artifacts_efficientnet_b5_flowers102"
PYTHON_SCRIPT="main.py"
NUM_GPUS=2

# Pass any extra arguments through to main.py (e.g. --use-wandb --wandb-api-key ...)
# Perforated Backpropagation: export PAIEMAIL + PAITOKEN and pass --dendrite-mode 2 (see mani/API/customization.md).
EXTRA_ARGS="$@"

echo "Step 1: Initializing PAI DDP settings (single GPU, exits after one batch)..."
python $PYTHON_SCRIPT --perforate_model_parallel $EXTRA_ARGS

echo ""
echo "Initialization complete. Starting continuous DDP training loop..."
echo "Press Ctrl+C to stop training."
echo ""

while true; do
    # Training already finished from a previous run
    if [ -f "${SAVE_NAME}/.training_complete" ]; then
        echo "Training already completed!"
        break
    fi

    if ls "${SAVE_NAME}"/switch_*.pt 1> /dev/null 2>&1; then
        echo "Resuming training from checkpoint..."
        torchrun --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT --pai_load_folder $SAVE_NAME $EXTRA_ARGS
    else
        echo "Starting training from beginning..."
        torchrun --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT $EXTRA_ARGS
    fi

    EXIT_CODE=$?

    if [ -f "${SAVE_NAME}/.training_complete" ]; then
        echo "Training completed successfully!"
        break
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Model restructured (dendrite added). Restarting in 2 seconds..."
        sleep 2
    else
        echo "Non-zero exit code ($EXIT_CODE). Stopping."
        exit $EXIT_CODE
    fi
done
