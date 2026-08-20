#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=upp-small
#SBATCH --output=upp-small-%j.out
#SBATCH --error=upp-small-%j.err

set -euo pipefail

jutil env activate -p reformo

MACHINE=$(cat /etc/FZJ/systemname)
USER_MACHINE="${USER}_${MACHINE}"

IMAGE="$FSCRATCH/$USER_MACHINE/containers/salt-arm64-astral-fa.sif"

WORK="$FSCRATCH/$USER_MACHINE/jetset-93940-preprocessing"

CONFIG="$WORK/configs/open-dataset-jupiter-small-test.yaml"

echo "============================================================"
echo "JETSET / UPP SMALL-DATASET TEST"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Arch:       $(uname -m)"
echo "Image:      $IMAGE"
echo "Config:     $CONFIG"
echo

test -f "$IMAGE"
test -f "$CONFIG"

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# Writable / headless matplotlib setup
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "=== Input data ==="
ls -lh \
  /e/data1/datasets/"$USER"/jupiter/atlas/jetset/93940/raw/mc-flavtag-ttbar-small.h5

echo
echo "=== UPP version ==="

srun apptainer exec \
    --cleanenv \
    --bind /e/data1:/e/data1 \
    --bind /e/fscratch:/e/fscratch \
    --env MPLBACKEND=Agg \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    "$IMAGE" \
    python - <<'PY'
import importlib.metadata as m
print("umami-preprocessing:", m.version("umami-preprocessing"))
PY

echo
echo "=== Starting UPP train preprocessing ==="

srun apptainer exec \
    --cleanenv \
    --bind /e/data1:/e/data1 \
    --bind /e/fscratch:/e/fscratch \
    --env MPLBACKEND=Agg \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    "$IMAGE" \
    preprocess \
        --config "$CONFIG" \
        --split train \
        --no-plot

echo
echo "============================================================"
echo "UPP SMALL TEST FINISHED"
echo "============================================================"

OUTPUT="$WORK/small-test/output"

echo
echo "=== Output ==="
find "$OUTPUT" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort

echo
echo "=== Expected SALT preprocessing products ==="

test -f "$OUTPUT/pp_output_train.h5"
test -f "$OUTPUT/norm_dict.yaml"
test -f "$OUTPUT/class_dict.yaml"

ls -lh \
    "$OUTPUT/pp_output_train.h5" \
    "$OUTPUT/norm_dict.yaml" \
    "$OUTPUT/class_dict.yaml"

echo
echo "============================================================"
echo "UPP SMALL TEST PASSED"
echo "============================================================"