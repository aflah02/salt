# GN2 Flash Attention 3 on JUPITER

This setup adds an explicit `flash3-varlen` backend without changing the existing
`flash-varlen` Flash Attention 2 backend. The FA3 backend is intentionally strict: it
raises instead of silently falling back when the wheel, Hopper GPU, dtype, cumulative
sequence lengths, head dimension, or zero-dropout requirement is not satisfied.

## Build the image

The definition installs the Astral aarch64 CUDA 13.0 / PyTorch 2.12 FA2 and FA3 wheels
side by side and clones `https://github.com/aflah02/salt.git` into `/salt`.

To build the PR branch before merging:

```bash
cd /e/scratch/reformo/${USER}_jupiter/salt

apptainer build \
    --fakeroot \
    --build-arg SALT_REF=feature/flash-attention-3 \
    "$FSCRATCH/${USER}_jupiter/containers/salt-arm64-astral-fa3.sif" \
    setup/salt-arm64-astral-fa3.def
```

After merging, omit `--build-arg SALT_REF=...`; the image defaults to the fork's `main`
branch. `/opt/SALT_COMMIT` records the exact commit baked into the image.

## Validate before the full run

Submit the four-GPU smoke job first:

```bash
sbatch GN2_slurm_configs/run_gn2_open_data_fa3_smoke.slurm
```

The smoke job verifies all four GPUs are SM90, checks the exact FA3 wheel, executes
FP16 and BF16 forward/backward passes through Salt with GN2's `D=256`, `H=8`, and
`S<=41` geometry, parses the FA3 config, and runs 20 training plus five validation
batches under four-process DDP.

After it passes, submit the 40-epoch run:

```bash
sbatch GN2_slurm_configs/run_gn2_open_data_fa3.slurm
```

Compare its full-step throughput and validation metrics with the existing FA2 run.
FA3 compatibility does not guarantee a speedup for GN2's short sequences and head
dimension 32.
