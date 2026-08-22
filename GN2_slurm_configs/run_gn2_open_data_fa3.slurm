#!/bin/bash

#SBATCH --account=reformo
#SBATCH --partition=booster

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4

#SBATCH --time=24:00:00
#SBATCH --job-name=gn2-fa3-full

#SBATCH --output=/e/scratch/reformo/khan27_jupiter/salt/logs/GN2_logs/gn2-fa3-full-%j.out
#SBATCH --error=/e/scratch/reformo/khan27_jupiter/salt/logs/GN2_logs/gn2-fa3-full-%j.err


set -euo pipefail


# ============================================================
# Environment
# ============================================================

jutil env activate -p reformo

MACHINE=$(cat /etc/FZJ/systemname)
USER_MACHINE="${USER}_${MACHINE}"


# ============================================================
# Paths
# ============================================================

REPO="/e/scratch/reformo/$USER_MACHINE/salt"

IMAGE="$FSCRATCH/$USER_MACHINE/containers/salt-arm64-astral-fa3.sif"

DATA="$FSCRATCH/$USER_MACHINE/jetset-93940-preprocessing/full/output"

RUNROOT="$FSCRATCH/$USER_MACHINE/salt-runs/GN2_open_data_FA3"

CONFIG="$REPO/salt/configs/GN2/GN2_open_data_jupiter_fa3.yaml"

LOG_DIR="$REPO/logs/GN2_logs"

RUN_NAME="GN2_open_data_FA3_full-${SLURM_JOB_ID}"

RUN_DIR="$RUNROOT/$RUN_NAME"

RESOLVED_CONFIG="$LOG_DIR/gn2-fa3-full-${SLURM_JOB_ID}-resolved.yaml"


mkdir -p "$RUNROOT"
mkdir -p "$LOG_DIR"


# Avoid each DDP process spawning large CPU BLAS thread pools.
# DataLoader parallelism is already controlled by num_workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


echo "============================================================"
echo "GN2 OPEN DATA FA3 - JUPITER 4-GPU FULL TRAINING"
echo "============================================================"

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Architecture:  $(uname -m)"
echo

echo "Slurm:"
echo "  Nodes:       $SLURM_JOB_NUM_NODES"
echo "  Tasks:       $SLURM_NTASKS"
echo "  CPUs/task:   $SLURM_CPUS_PER_TASK"
echo

echo "Image:"
echo "  $IMAGE"
echo

echo "Config:"
echo "  $CONFIG"
echo

echo "Dataset:"
echo "  $DATA"
echo

echo "Run root:"
echo "  $RUNROOT"
echo

echo "Expected run directory:"
echo "  $RUN_DIR"
echo

echo "Slurm logs:"
echo "  $LOG_DIR"
echo

echo "============================================================"
echo


# ============================================================
# Preflight: required files
# ============================================================

echo "=== Preflight ==="

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

echo "All required files present."
echo


# ============================================================
# Host GPU allocation
# ============================================================

echo "=== Allocated GPUs ==="

nvidia-smi -L

GPU_COUNT=$(nvidia-smi -L | wc -l)

echo
echo "Visible host GPUs: $GPU_COUNT"

if [[ "$GPU_COUNT" -ne 4 ]]; then
    echo "ERROR: expected 4 GPUs on the allocated node."
    exit 1
fi

echo


# ============================================================
# Container/software check
#
# One Slurm task owns all four allocated GPUs.
# ============================================================

echo "=== Container software / GPU visibility ==="

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
        python - <<'PY'

import platform
import torch
import salt
import salt.models

print("Architecture:", platform.machine())
print("SALT:", salt.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

assert platform.machine() == "aarch64"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert hasattr(salt.models, "Transformer")

print()
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print()
print("Transformer implementation:")
print(salt.models.Transformer)

print()
print("4-GPU container check: SUCCESS")

PY

echo


# ============================================================
# Execute the FA3 kernel through Salt with GN2 dimensions
# ============================================================

echo "=== Flash Attention 3 GN2 preflight ==="

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
        python /salt/GN2_slurm_configs/check_fa3_gn2.py

echo


# ============================================================
# Parse the adapted config with SALT inside the image
#
# This catches stale class paths / argument syntax before we
# launch four distributed processes.
# ============================================================

echo "=== Parsing adapted SALT config ==="

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
            --print_config \
        > "$RESOLVED_CONFIG"

echo "SALT config parse: SUCCESS"
echo "Resolved config:"
echo "  $RESOLVED_CONFIG"
echo


# ============================================================
# Distributed full training
#
# torchrun launches four SALT processes inside the SIF:
#
#   local rank 0 -> GPU 0
#   local rank 1 -> GPU 1
#   local rank 2 -> GPU 2
#   local rank 3 -> GPU 3
#
# SALT/Lightning then uses DDP across the four local GPUs.
#
# We are NOT running host-clone Python code.
# The config merely lives in the clone.
# ============================================================

echo "============================================================"
echo "STARTING 4-GPU SALT FULL TRAINING"
echo "============================================================"

echo "Nodes:                  1"
echo "GPUs/node:              4"
echo "DDP processes:          4"
echo
echo "Epochs:                 40 (from config)"
echo "Train batches:          all"
echo "Validation batches:     all"
echo
echo "Batch size / GPU:       3000"
echo "Effective global batch: 12000"
echo
echo "Expected run directory:"
echo "  $RUN_DIR"
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
            --log_suffix "full-${SLURM_JOB_ID}" \
            --trainer.num_nodes=1 \
            --trainer.devices=4 \
            --trainer.strategy=ddp


END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))


echo
echo "============================================================"
echo "GN2 OPEN DATA FA3 4-GPU FULL TRAINING PASSED"
echo "============================================================"
echo
echo "Training elapsed seconds: $ELAPSED"
echo


# ============================================================
# Check run outputs
# ============================================================

echo "=== Run directory ==="

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: expected SALT run directory was not created:"
    echo "  $RUN_DIR"
    exit 1
fi

find "$RUN_DIR" \
    -maxdepth 3 \
    -type f \
    -printf '%p %s bytes\n' \
    | sort

echo


# ============================================================
# Check checkpoints
# ============================================================

echo "=== Checkpoints ==="

CKPT_DIR="$RUN_DIR/ckpts"

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: checkpoint directory not found:"
    echo "  $CKPT_DIR"
    exit 1
fi

find "$CKPT_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.ckpt' \
    -printf '%p %s bytes\n' \
    | sort

CKPT_COUNT=$(find "$CKPT_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.ckpt' \
    | wc -l)

if [[ "$CKPT_COUNT" -lt 1 ]]; then
    echo "ERROR: no checkpoint was written."
    exit 1
fi

echo
echo "Checkpoint count: $CKPT_COUNT"
echo


# ============================================================
# Final result
# ============================================================

echo "============================================================"
echo "ALL GN2 FA3 4-GPU TRAINING CHECKS PASSED"
echo "============================================================"
echo
echo "Slurm stdout:"
echo "  $LOG_DIR/gn2-fa3-full-${SLURM_JOB_ID}.out"
echo
echo "Slurm stderr:"
echo "  $LOG_DIR/gn2-fa3-full-${SLURM_JOB_ID}.err"
echo
echo "Resolved config:"
echo "  $RESOLVED_CONFIG"
echo
echo "Checkpoints:"
echo "  $CKPT_DIR"
echo "============================================================"
