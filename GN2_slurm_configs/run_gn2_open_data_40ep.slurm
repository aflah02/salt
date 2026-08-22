#!/bin/bash

#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=gn2-open-40ep

#SBATCH --output=/e/scratch/reformo/khan27_jupiter/salt/logs/GN2_logs/gn2-40ep-%j.out
#SBATCH --error=/e/scratch/reformo/khan27_jupiter/salt/logs/GN2_logs/gn2-40ep-%j.err


set -euo pipefail


# ============================================================
# Environment
# ============================================================

jutil env activate -p reformo

REPO="/e/scratch/reformo/khan27_jupiter/salt"

IMAGE="/e/fscratch/reformo/khan27_jupiter/containers/salt-arm64-astral-fa.sif"

DATA="/e/fscratch/reformo/khan27_jupiter/jetset-93940-preprocessing/full/output"

RUNROOT="/e/fscratch/reformo/khan27_jupiter/salt-runs/GN2_open_data"

CONFIG="$REPO/salt/configs/GN2/GN2_open_data_jupiter.yaml"

LOG_DIR="$REPO/logs/GN2_logs"

RUN_NAME="GN2_open_data_prod-${SLURM_JOB_ID}"

RUN_DIR="$RUNROOT/$RUN_NAME"

RESOLVED_CONFIG="$LOG_DIR/gn2-40ep-${SLURM_JOB_ID}-resolved.yaml"


mkdir -p "$RUNROOT"
mkdir -p "$LOG_DIR"


# ============================================================
# Comet offline logging
#
# Keep API key visible to SALT so SALT does not inject
# online=False into Lightning's CometLogger.
#
# Tell the Comet SDK itself to start offline.
# SALT will place the offline archive in RUN_DIR.
# ============================================================

source "$HOME/.config/comet/env"

export APPTAINERENV_COMET_API_KEY="$COMET_API_KEY"
export APPTAINERENV_COMET_START_ONLINE=0

# Do not force a separate offline directory. SALT sets this to
# the generated run directory itself.
unset APPTAINERENV_COMET_OFFLINE_DIRECTORY || true


# ============================================================
# CPU threading
# ============================================================

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Current JUPITER Slurm does not automatically propagate
# --cpus-per-task from sbatch to srun.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"


# ============================================================
# Run information
# ============================================================

echo "============================================================"
echo "GN2 OPEN DATA - 40 EPOCH PRODUCTION RUN"
echo "============================================================"
echo
echo "Job ID:              $SLURM_JOB_ID"
echo "Node:                $(hostname)"
echo "Architecture:        $(uname -m)"
echo
echo "GPUs:                4"
echo "Batch size / GPU:    3000"
echo "Global batch size:   12000"
echo "Epochs:              40"
echo
echo "Training jets:       53,500,000"
echo "Validation jets:     5,350,000"
echo
echo "Image:"
echo "  $IMAGE"
echo
echo "Config:"
echo "  $CONFIG"
echo
echo "Run directory:"
echo "  $RUN_DIR"
echo
echo "============================================================"
echo


# ============================================================
# Required files
# ============================================================

for FILE in \
    "$IMAGE" \
    "$CONFIG" \
    "$DATA/pp_output_train.h5" \
    "$DATA/pp_output_val.h5" \
    "$DATA/norm_dict.yaml" \
    "$DATA/class_dict.yaml"
do
    if [[ ! -f "$FILE" ]]; then
        echo "ERROR: required file missing:"
        echo "  $FILE"
        exit 1
    fi
done

echo "Required-file check: SUCCESS"
echo


# ============================================================
# GPU/container sanity check
# ============================================================

echo "=== GPU / container check ==="

nvidia-smi -L

srun \
    --ntasks=1 \
    --gres=gpu:4 \
    apptainer exec \
        --nv \
        --cleanenv \
        --pwd /salt \
        --env PYTHONPATH=/salt \
        --bind /e/scratch:/e/scratch \
        --bind /e/fscratch:/e/fscratch \
        "$IMAGE" \
        python -c '
import torch
import salt

print("SALT:", salt.__version__)
print("SALT path:", salt.__file__)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert salt.__file__.startswith("/salt/")
'

echo
echo "GPU/container check: SUCCESS"
echo


# ============================================================
# Resolve exact production configuration
# ============================================================

echo "=== Resolving production SALT config ==="

srun \
    --ntasks=1 \
    --gres=gpu:4 \
    apptainer exec \
        --nv \
        --cleanenv \
        --pwd /salt \
        --env PYTHONPATH=/salt \
        --bind /e/scratch:/e/scratch \
        --bind /e/fscratch:/e/fscratch \
        "$IMAGE" \
        salt fit \
            --config "$CONFIG" \
            --force \
            --log_suffix "prod-${SLURM_JOB_ID}" \
            --trainer.num_nodes=1 \
            --trainer.devices=4 \
            --trainer.strategy=ddp \
            --trainer.max_epochs=40 \
            --print_config \
        > "$RESOLVED_CONFIG"

echo "Resolved config:"
echo "  $RESOLVED_CONFIG"
echo


# ============================================================
# Full 40 epoch training
#
# One Slurm task owns the complete node and all four GPUs.
# torchrun creates four local DDP ranks:
#
# rank 0 -> GPU 0
# rank 1 -> GPU 1
# rank 2 -> GPU 2
# rank 3 -> GPU 3
#
# SALT batch_size=3000 is per rank:
#
# 3000 x 4 = 12000 global batch.
# ============================================================

echo "============================================================"
echo "STARTING GN2 TRAINING"
echo "============================================================"
echo

START_TIME=$(date +%s)

srun \
    --ntasks=1 \
    --gres=gpu:4 \
    apptainer exec \
        --nv \
        --cleanenv \
        --pwd /salt \
        --env PYTHONPATH=/salt \
        --bind /e/scratch:/e/scratch \
        --bind /e/fscratch:/e/fscratch \
        "$IMAGE" \
        python -m torch.distributed.run \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=4 \
            -m salt.main \
            fit \
            --config "$CONFIG" \
            --force \
            --log_suffix "prod-${SLURM_JOB_ID}" \
            --trainer.num_nodes=1 \
            --trainer.devices=4 \
            --trainer.strategy=ddp \
            --trainer.max_epochs=40

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))


# ============================================================
# Results
# ============================================================

echo
echo "============================================================"
echo "GN2 40 EPOCH TRAINING FINISHED"
echo "============================================================"
echo
echo "Elapsed seconds: $ELAPSED"
echo "Run directory:"
echo "  $RUN_DIR"
echo


if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: expected run directory not found:"
    echo "  $RUN_DIR"
    exit 1
fi


echo "=== Checkpoints ==="

CKPT_DIR="$RUN_DIR/ckpts"

if [[ -d "$CKPT_DIR" ]]; then
    find "$CKPT_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.ckpt' \
        -printf '%f  %s bytes\n' \
        | sort

    CKPT_COUNT=$(
        find "$CKPT_DIR" \
            -maxdepth 1 \
            -type f \
            -name '*.ckpt' \
        | wc -l
    )

    echo
    echo "Checkpoint count: $CKPT_COUNT"
else
    echo "WARNING: checkpoint directory not found."
fi


echo
echo "=== Offline Comet archives ==="

find "$RUN_DIR" \
    -maxdepth 2 \
    -type f \
    -name '*.zip' \
    -printf '%p  %s bytes\n' \
    | sort


echo
echo "============================================================"
echo "OUTPUTS"
echo "============================================================"
echo
echo "Run:"
echo "  $RUN_DIR"
echo
echo "Checkpoints:"
echo "  $RUN_DIR/ckpts"
echo
echo "Resolved config:"
echo "  $RESOLVED_CONFIG"
echo
echo "Slurm stdout:"
echo "  $LOG_DIR/gn2-40ep-${SLURM_JOB_ID}.out"
echo
echo "Slurm stderr:"
echo "  $LOG_DIR/gn2-40ep-${SLURM_JOB_ID}.err"
echo
echo "After the job, upload Comet archives from the login node with:"
echo "  $REPO/GN2_slurm_configs/upload_all_comet_offline.sh"
echo
echo "============================================================"