#!/bin/bash

set -euo pipefail

DATA="/e/data1/datasets/${USER}/jupiter/atlas/jetset/93940/raw"

declare -A EXPECTED_SIZE
declare -A EXPECTED_ADLER

EXPECTED_SIZE["mc-flavtag-ttbar-small.h5"]="3055139850"
EXPECTED_SIZE["mc-flavtag-ttbar-medium.h5"]="13935881976"
EXPECTED_SIZE["mc-flavtag-ttbar-large.h5"]="90797951148"

EXPECTED_ADLER["mc-flavtag-ttbar-small.h5"]="8d7e9098"
EXPECTED_ADLER["mc-flavtag-ttbar-medium.h5"]="e29be59c"
EXPECTED_ADLER["mc-flavtag-ttbar-large.h5"]="61adcbcb"

FILES=(
    "mc-flavtag-ttbar-small.h5"
    "mc-flavtag-ttbar-medium.h5"
    "mc-flavtag-ttbar-large.h5"
)

echo "============================================================"
echo "ATLAS JetSet record 93940 verification"
echo "============================================================"
echo "Directory: $DATA"
echo

for FILE in "${FILES[@]}"; do
    PATH_TO_FILE="$DATA/$FILE"

    echo "------------------------------------------------------------"
    echo "$FILE"
    echo "------------------------------------------------------------"

    if [[ ! -f "$PATH_TO_FILE" ]]; then
        echo "ERROR: file not found:"
        echo "  $PATH_TO_FILE"
        exit 1
    fi

    ACTUAL_SIZE=$(stat -c %s "$PATH_TO_FILE")

    echo "Size:"
    echo "  actual:   $ACTUAL_SIZE"
    echo "  expected: ${EXPECTED_SIZE[$FILE]}"

    if [[ "$ACTUAL_SIZE" != "${EXPECTED_SIZE[$FILE]}" ]]; then
        echo "ERROR: size mismatch"
        exit 1
    fi

    echo "  SIZE OK"
    echo

    echo "Computing Adler-32..."

    ACTUAL_ADLER=$(
        python3 - "$PATH_TO_FILE" <<'PY'
import sys
import zlib

path = sys.argv[1]

checksum = 1

with open(path, "rb") as f:
    while True:
        chunk = f.read(64 * 1024 * 1024)
        if not chunk:
            break
        checksum = zlib.adler32(chunk, checksum)

print(f"{checksum & 0xffffffff:08x}")
PY
    )

    echo "Adler-32:"
    echo "  actual:   $ACTUAL_ADLER"
    echo "  expected: ${EXPECTED_ADLER[$FILE]}"

    if [[ "$ACTUAL_ADLER" != "${EXPECTED_ADLER[$FILE]}" ]]; then
        echo "ERROR: Adler-32 mismatch"
        exit 1
    fi

    echo "  CHECKSUM OK"
    echo
done

echo "============================================================"
echo "ALL JETSET FILES VERIFIED SUCCESSFULLY"
echo "============================================================"

echo
ls -lh "$DATA"
echo
du -sh "$DATA"