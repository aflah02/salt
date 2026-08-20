#!/bin/bash

set -euo pipefail

# ------------------------------------------------------------
# ATLAS JetSet
# CERN Open Data record 93940
# ------------------------------------------------------------

jutil env activate -p datasets

DATASET_ROOT="/e/data1/datasets/${USER}/jupiter/atlas/jetset/93940"
RAW_DIR="${DATASET_ROOT}/raw"
META_DIR="${DATASET_ROOT}/metadata"

mkdir -p "$RAW_DIR" "$META_DIR"

BASE_URL="https://opendata.cern.ch/record/93940/files"

FILES=(
    "mc-flavtag-ttbar-small.h5"
    "mc-flavtag-ttbar-medium.h5"
    "mc-flavtag-ttbar-large.h5"
)

SIZES=(
    "3055139850"
    "13935881976"
    "90797951148"
)

ADLER32=(
    "8d7e9098"
    "e29be59c"
    "61adcbcb"
)

echo "============================================================"
echo "ATLAS JetSet / CERN Open Data record 93940"
echo "============================================================"
echo "Destination: $RAW_DIR"
echo

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    EXPECTED_SIZE="${SIZES[$i]}"
    EXPECTED_ADLER="${ADLER32[$i]}"

    URL="${BASE_URL}/${FILE}"
    DEST="${RAW_DIR}/${FILE}"

    echo "============================================================"
    echo "Downloading $FILE"
    echo "============================================================"
    echo "URL:            $URL"
    echo "Destination:    $DEST"
    echo "Expected bytes: $EXPECTED_SIZE"
    echo "Adler-32:       $EXPECTED_ADLER"
    echo

    # -C - resumes an interrupted download.
    curl \
        --fail \
        --location \
        --continue-at - \
        --retry 10 \
        --retry-delay 10 \
        --retry-all-errors \
        --output "$DEST" \
        "$URL"

    echo
    echo "Download finished:"
    ls -lh "$DEST"

    ACTUAL_SIZE=$(stat -c %s "$DEST")

    if [[ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]]; then
        echo "ERROR: size mismatch for $FILE"
        echo "Expected: $EXPECTED_SIZE"
        echo "Actual:   $ACTUAL_SIZE"
        exit 1
    fi

    echo "Size check: OK"
    echo
done

echo "============================================================"
echo "ALL DOWNLOADS COMPLETE"
echo "============================================================"

du -sh "$RAW_DIR"
ls -lh "$RAW_DIR"