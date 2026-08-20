#!/bin/bash
set -euo pipefail

jutil env activate -p reformo

MACHINE=$(cat /etc/FZJ/systemname)
USER_MACHINE="${USER}_${MACHINE}"

IMAGE="$FSCRATCH/$USER_MACHINE/containers/salt-arm64-astral-fa.sif"

OUTPUT="$FSCRATCH/$USER_MACHINE/jetset-93940-preprocessing/small-test/output"
H5="$OUTPUT/pp_output_train.h5"

echo "============================================================"
echo "CHECKING UPP OUTPUT FOR SALT GN2"
echo "============================================================"
echo "File: $H5"
echo

apptainer exec \
    --cleanenv \
    --bind /e/fscratch:/e/fscratch \
    "$IMAGE" \
    python - "$H5" <<'PY'
import sys
import h5py

path = sys.argv[1]

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

with h5py.File(path, "r") as f:

    print("Top-level datasets:")
    for name in f:
        print(f"  {name}: shape={f[name].shape}")
    print()

    assert "jets" in f, "Missing jets dataset"
    assert "tracks" in f, "Missing tracks dataset"

    jet_fields = set(f["jets"].dtype.names or [])
    track_fields = set(f["tracks"].dtype.names or [])

    print("Jet fields:")
    for x in sorted(jet_fields):
        print(" ", x)

    print("\nTrack fields:")
    for x in sorted(track_fields):
        print(" ", x)

    missing_jets = required_jets - jet_fields
    missing_tracks = required_tracks - track_fields

    print()
    print("Missing required jet fields:", sorted(missing_jets))
    print("Missing required track fields:", sorted(missing_tracks))

    if missing_jets or missing_tracks:
        raise SystemExit("ERROR: output is not compatible with GN2_open_data.yaml")

    print()
    print("Jet shape:  ", f["jets"].shape)
    print("Track shape:", f["tracks"].shape)

print()
print("ALL REQUIRED GN2 VARIABLES ARE PRESENT")
PY

echo
echo "============================================================"
echo "NORMALISATION DICTIONARY"
echo "============================================================"
cat "$OUTPUT/norm_dict.yaml"

echo
echo "============================================================"
echo "CLASS DICTIONARY"
echo "============================================================"
cat "$OUTPUT/class_dict.yaml"

echo
echo "============================================================"
echo "CHECK COMPLETE"
echo "============================================================"