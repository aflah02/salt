#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=salt-arm64-test
#SBATCH --output=salt-arm64-test-%j.out

set -euo pipefail

jutil env activate -p reformo

MACHINE=$(cat /etc/FZJ/systemname)
USER_MACHINE="${USER}_${MACHINE}"

IMAGE="$FSCRATCH/$USER_MACHINE/containers/salt-arm64-astral-fa.sif"

echo "========================================"
echo "SALT ARM64 GPU TEST"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Host architecture: $(uname -m)"
echo "Image: $IMAGE"
echo

test -f "$IMAGE"

echo "=== Host GPU ==="
nvidia-smi -L
echo

srun apptainer exec \
    --nv \
    --cleanenv \
    --bind "$SCRATCH:$SCRATCH" \
    --bind "$FSCRATCH:$FSCRATCH" \
    "$IMAGE" \
    python - <<'PY'
import platform
import torch
import flash_attn
import salt

print("========================================")
print("CONTAINER")
print("========================================")

print("Architecture:", platform.machine())
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("FlashAttention:", flash_attn.__version__)
print("SALT:", salt.__version__)
print()

assert platform.machine() == "aarch64", platform.machine()
assert torch.__version__.startswith("2.12.1"), torch.__version__
assert torch.version.cuda is not None
assert torch.cuda.is_available(), "CUDA is not available inside container"
assert torch.cuda.device_count() >= 1

print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
print()

print("=== Basic CUDA tensor test ===")

x = torch.randn(1024, 1024, device="cuda")
y = x @ x

torch.cuda.synchronize()

print("Result shape:", y.shape)
print("Device:", y.device)
print("GPU tensor test: SUCCESS")

print()
print("========================================")
print("SALT ARM64 GPU TEST PASSED")
print("========================================")
PY
