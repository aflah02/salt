#!/usr/bin/env bash
set -uo pipefail

# Upload every Comet offline ZIP found under the GN2 Open Data run root.
# Run this from a JUPITER login node (compute nodes do not have outbound internet).

IMAGE="/e/fscratch/reformo/khan27_jupiter/containers/salt-arm64-astral-fa.sif"
RUNROOT="/e/fscratch/reformo/khan27_jupiter/salt-runs/GN2_open_data"
COMET_ENV="$HOME/.config/comet/env"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: Apptainer image not found:"
    echo "  $IMAGE"
    exit 1
fi

if [[ ! -d "$RUNROOT" ]]; then
    echo "ERROR: Run root not found:"
    echo "  $RUNROOT"
    exit 1
fi

if [[ ! -f "$COMET_ENV" ]]; then
    echo "ERROR: Comet credential file not found:"
    echo "  $COMET_ENV"
    exit 1
fi

# Load COMET_API_KEY without putting it in this script.
# shellcheck disable=SC1090
source "$COMET_ENV"

if [[ -z "${COMET_API_KEY:-}" ]]; then
    echo "ERROR: COMET_API_KEY is not set by $COMET_ENV"
    exit 1
fi

# Pass the API key into the clean Apptainer environment.
export APPTAINERENV_COMET_API_KEY="$COMET_API_KEY"

# Uploading must be online. Do not inherit the variables used during offline training.
unset APPTAINERENV_COMET_START_ONLINE || true
unset APPTAINERENV_COMET_OFFLINE_DIRECTORY || true
unset COMET_START_ONLINE || true
unset COMET_OFFLINE_DIRECTORY || true

mapfile -d '' ZIPS < <(
    find "$RUNROOT" \
        -type f \
        -name '*.zip' \
        -print0 \
    | sort -z
)

TOTAL="${#ZIPS[@]}"

if (( TOTAL == 0 )); then
    echo "No offline Comet ZIP files found under:"
    echo "  $RUNROOT"
    exit 0
fi

echo "Found $TOTAL offline Comet ZIP file(s)."
echo

SUCCESS=0
FAILED=0

for ZIP in "${ZIPS[@]}"; do
    echo "============================================================"
    echo "Uploading:"
    echo "  $ZIP"
    echo "============================================================"

    if apptainer exec \
        --cleanenv \
        --bind /e/fscratch:/e/fscratch \
        "$IMAGE" \
        comet upload "$ZIP"
    then
        ((SUCCESS+=1))
        echo "SUCCESS: $ZIP"
    else
        ((FAILED+=1))
        echo "FAILED:  $ZIP" >&2
    fi

    echo
done

echo "============================================================"
echo "Comet offline upload summary"
echo "============================================================"
echo "Found:      $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed:     $FAILED"
echo "============================================================"

if (( FAILED > 0 )); then
    exit 1
fi