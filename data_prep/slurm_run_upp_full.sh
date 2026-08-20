#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=upp-jetset-full
#SBATCH --output=upp-jetset-full-%j.out
#SBATCH --error=upp-jetset-full-%j.err

set -euo pipefail

# ============================================================
# Environment
# ============================================================

jutil env activate -p reformo

MACHINE=$(cat /etc/FZJ/systemname)
USER_MACHINE="${USER}_${MACHINE}"

IMAGE="$FSCRATCH/$USER_MACHINE/containers/salt-arm64-astral-fa.sif"

WORK="$FSCRATCH/$USER_MACHINE/jetset-93940-preprocessing"

CONFIG="$WORK/configs/open-dataset-jupiter-full.yaml"

RAW="/e/data1/datasets/$USER/jupiter/atlas/jetset/93940/raw"

OUTPUT="$WORK/full/output"

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# Prevent GUI / matplotlib issues on compute nodes
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/cache-${USER}-${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "ATLAS JETSET FULL UPP PREPROCESSING"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Architecture: $(uname -m)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo
echo "Image:"
echo "  $IMAGE"
echo
echo "Config:"
echo "  $CONFIG"
echo
echo "Raw data:"
echo "  $RAW"
echo
echo "Output:"
echo "  $OUTPUT"
echo "============================================================"
echo


# ============================================================
# Preflight checks
# ============================================================

echo "=== Checking image ==="

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: Container image does not exist:"
    echo "$IMAGE"
    exit 1
fi

echo "Image: OK"
echo


echo "=== Checking config ==="

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Full preprocessing config does not exist:"
    echo "$CONFIG"
    exit 1
fi

echo "Config: OK"
echo


echo "=== Checking CERN JetSet files ==="

FILES=(
    "mc-flavtag-ttbar-small.h5"
    "mc-flavtag-ttbar-medium.h5"
    "mc-flavtag-ttbar-large.h5"
)

for FILE in "${FILES[@]}"; do
    if [[ ! -f "$RAW/$FILE" ]]; then
        echo "ERROR: Missing input:"
        echo "$RAW/$FILE"
        exit 1
    fi

    ls -lh "$RAW/$FILE"
done

echo
echo "All three input files present."
echo


# ============================================================
# Guard against accidentally running smoke-test config
# ============================================================

echo "=== Checking that this is the FULL config ==="

if ! grep -q 'mc-flavtag-ttbar-\*\.h5' "$CONFIG"; then
    echo "ERROR:"
    echo "Config does not contain the expected full-dataset glob:"
    echo '  mc-flavtag-ttbar-*.h5'
    exit 1
fi

if grep -q "small-test" "$CONFIG"; then
    echo "ERROR:"
    echo "Config still contains 'small-test'."
    exit 1
fi

if ! grep -q \
    '/e/fscratch/reformo/khan27_jupiter/jetset-93940-preprocessing/full' \
    "$CONFIG"; then
    echo "ERROR:"
    echo "Config does not point to the full preprocessing directory."
    exit 1
fi

echo "Full dataset config check: OK"
echo

# ============================================================
# Container / UPP check
# ============================================================

echo "=== Software versions ==="

srun apptainer exec \
    --cleanenv \
    --bind /e/data1:/e/data1 \
    --bind /e/fscratch:/e/fscratch \
    --env MPLBACKEND=Agg \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    "$IMAGE" \
    python - <<'PY'
import platform
import importlib.metadata as metadata

print("Architecture:", platform.machine())
print("Python UPP:", metadata.version("umami-preprocessing"))

assert metadata.version("umami-preprocessing") == "0.3.1"
print("UPP version check: OK")
PY

echo


# ============================================================
# Parse config before doing expensive work
# ============================================================

echo "=== Parsing preprocessing config ==="

srun apptainer exec \
    --cleanenv \
    --bind /e/data1:/e/data1 \
    --bind /e/fscratch:/e/fscratch \
    --env MPLBACKEND=Agg \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    "$IMAGE" \
    python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from upp.classes.preprocessing_config import PreprocessingConfig

config_path = Path(sys.argv[1])

cfg = PreprocessingConfig.from_file(
    config_path,
    split="train",
)

print()
print("CONFIG PARSED SUCCESSFULLY")
print("base_dir:   ", cfg.base_dir)
print("ntuple_dir: ", cfg.ntuple_dir)
print("out_fname:  ", cfg.out_fname)

print()
print("Components:")

for component in cfg.components:
    print(
        f"  {component.name:<25} "
        f"{component.num_jets:,} jets"
    )

print()
print("Total requested train jets:",
      f"{sum(c.num_jets for c in cfg.components):,}")
PY

echo


# ============================================================
# Start full preprocessing
# ============================================================

echo "============================================================"
echo "STARTING FULL UPP PREPROCESSING"
echo "Splits: train + val + test"
echo "Plotting: disabled"
echo "============================================================"
echo

START_TIME=$(date +%s)

srun apptainer exec \
    --cleanenv \
    --bind /e/data1:/e/data1 \
    --bind /e/fscratch:/e/fscratch \
    --env MPLBACKEND=Agg \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    "$IMAGE" \
    preprocess \
        --config "$CONFIG" \
        --split all \
        --no-plot

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo "============================================================"
echo "UPP FULL PREPROCESSING FINISHED"
echo "Elapsed seconds: $ELAPSED"
echo "============================================================"
echo


# ============================================================
# Check output
# ============================================================

echo "=== Output files ==="

find "$OUTPUT" \
    -maxdepth 2 \
    -type f \
    -printf '%p %s bytes\n' \
    | sort

echo


# ============================================================
# Required SALT outputs
# ============================================================

echo "=== Checking SALT-required outputs ==="

REQUIRED=(
    "$OUTPUT/pp_output_train.h5"
    "$OUTPUT/pp_output_val.h5"
    "$OUTPUT/norm_dict.yaml"
    "$OUTPUT/class_dict.yaml"
)

for FILE in "${REQUIRED[@]}"; do
    if [[ ! -f "$FILE" ]]; then
        echo "ERROR: Missing required output:"
        echo "$FILE"
        exit 1
    fi

    ls -lh "$FILE"
done

echo


# ============================================================
# Test output is also expected because --split all was used
# ============================================================

echo "=== Checking test output ==="

TEST_FILES=("$OUTPUT"/pp_output_test*.h5)

if (( ${#TEST_FILES[@]} > 0 )) && [[ -e "${TEST_FILES[0]}" ]]; then
    ls -lh "${TEST_FILES[@]}"
else
    echo "ERROR: No test output found."
    exit 1
fi

echo


# ============================================================
# Basic HDF5 integrity / SALT schema check
# ============================================================

echo "=== Checking processed HDF5 schema ==="

srun apptainer exec \
    --cleanenv \
    --bind /e/fscratch:/e/fscratch \
    "$IMAGE" \
    python - "$OUTPUT" <<'PY'
import sys
from pathlib import Path

import h5py

output = Path(sys.argv[1])

required_jets = {
    "pt_btagJes",
    "eta_btagJes",
    "flavour_label",
}

required_tracks = {
    "d0",
    "z0SinTheta",
    "dphi",
    "deta",
    "qOverP",
    "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinThetaSignificance",
    "phiUncertainty",
    "thetaUncertainty",
    "qOverPUncertainty",
    "numberOfPixelHits",
    "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits",
    "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits",
    "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits",
    "numberOfPixelSplitHits",
    "numberOfSCTSharedHits",
    "ftagTruthOriginLabel",
    "ftagTruthVertexIndex",
}

for split in ("train", "val", "test"):
    path = output / f"pp_output_{split}.h5"

    if not path.exists():
        if split == "test":
            continue
        raise RuntimeError(f"Missing {path}")

    print()
    print("=" * 70)
    print(path)
    print("=" * 70)

    with h5py.File(path, "r") as f:
        assert "jets" in f
        assert "tracks" in f

        print("jets:  ", f["jets"].shape)
        print("tracks:", f["tracks"].shape)

        jet_fields = set(f["jets"].dtype.names or ())
        track_fields = set(f["tracks"].dtype.names or ())

        missing_jets = required_jets - jet_fields
        missing_tracks = required_tracks - track_fields

        if missing_jets:
            raise RuntimeError(
                f"{split}: missing jet variables: {sorted(missing_jets)}"
            )

        if missing_tracks:
            raise RuntimeError(
                f"{split}: missing track variables: {sorted(missing_tracks)}"
            )

        print("GN2 variable check: OK")

print()
print("ALL PROCESSED HDF5 FILES PASSED")
PY

echo


# ============================================================
# Final summary
# ============================================================

echo "============================================================"
echo "FULL JETSET PREPROCESSING PASSED"
echo "============================================================"

echo
echo "Training:"
ls -lh "$OUTPUT/pp_output_train.h5"

echo
echo "Validation:"
ls -lh "$OUTPUT/pp_output_val.h5"

echo
echo "Normalisation:"
ls -lh "$OUTPUT/norm_dict.yaml"

echo
echo "Classes:"
ls -lh "$OUTPUT/class_dict.yaml"

if [[ -f "$OUTPUT/pp_output_test.h5" ]]; then
    echo
    echo "Test:"
    ls -lh "$OUTPUT/pp_output_test.h5"
fi

echo
echo "These files are ready to be referenced from GN2_open_data.yaml."
echo