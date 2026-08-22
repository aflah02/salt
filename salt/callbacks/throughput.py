"""End-to-end training throughput logging."""

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer


def get_batch_size(batch: Any) -> int:
    """Return the leading dimension of the first tensor in a nested batch."""
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            raise ValueError("Cannot infer batch size from a scalar tensor")
        return batch.shape[0]

    values: Any
    if isinstance(batch, Mapping):
        values = batch.values()
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        values = batch
    else:
        raise TypeError(f"Cannot infer batch size from {type(batch).__name__}")

    for value in values:
        try:
            return get_batch_size(value)
        except (TypeError, ValueError):
            continue
    raise ValueError("Cannot infer batch size because the batch contains no non-scalar tensors")


class ThroughputLogger(Callback):
    """Log rolling, end-to-end training throughput on rank zero.

    Timing covers data loading, forward, backward, optimizer work, and DDP synchronization.
    CUDA is synchronized only at measurement boundaries, avoiding a synchronization on every
    batch. Metrics are sent to the configured Lightning logger, printed to stdout, and written
    to ``throughput.jsonl`` so they remain available when ``trainer.logger`` is disabled.

    Parameters
    ----------
    log_every_n_steps : int, optional
        Number of dataloader batches per device in each measurement window. By default 50.
    warmup_steps : int, optional
        Initial dataloader batches to exclude from timing. By default 10.
    dir_path : str | None, optional
        Directory for ``throughput.jsonl``. Uses ``trainer.log_dir`` by default.
    std_out : bool, optional
        Print every measurement on rank zero. By default True.

    Notes
    -----
    ``samples_per_sec`` and ``batches_per_sec`` are global aggregate rates. For example, four
    DDP ranks completing one local batch count as four global batches. ``steps_per_sec`` counts
    synchronized dataloader steps and is therefore independent of world size.
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        warmup_steps: int = 10,
        dir_path: str | None = None,
        std_out: bool = True,
    ) -> None:
        super().__init__()
        if log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be at least 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")

        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.dir_path = dir_path
        self.std_out = std_out

        self.path: Path | None = None
        self._active = False
        self._writer = False
        self._warmup_remaining = warmup_steps
        self._window_start: float | None = None
        self._window_samples = 0
        self._window_batches = 0

    def setup(self, trainer: Trainer, _module: LightningModule, stage: str) -> None:
        """Create the rank-zero JSON-lines output file for fitting."""
        self._active = stage == "fit" and not trainer.fast_dev_run
        self._writer = self._active and trainer.is_global_zero
        if not self._active:
            return

        # trainer.log_dir performs a distributed broadcast, so every rank
        # must evaluate it in the same collective order.
        out_dir = Path(trainer.log_dir) if self.dir_path is None else Path(self.dir_path)

        if not self._writer:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "throughput.jsonl"
        self.path.write_text("")

    @staticmethod
    def _synchronize(trainer: Trainer, name: str) -> None:
        root_device = getattr(trainer.strategy, "root_device", None)
        if root_device is not None and root_device.type == "cuda":
            torch.cuda.synchronize(root_device)
        barrier = getattr(trainer.strategy, "barrier", None)
        if trainer.world_size > 1 and barrier is not None:
            barrier(name)

    def _start_window(self, trainer: Trainer) -> None:
        self._synchronize(trainer, "throughput_window_start")
        self._window_start = time.perf_counter()
        self._window_samples = 0
        self._window_batches = 0

    def on_train_start(self, trainer: Trainer, _module: LightningModule) -> None:
        """Start timing immediately when no warmup was requested."""
        if self._active and self._warmup_remaining == 0:
            self._start_window(trainer)

    def on_train_epoch_start(self, trainer: Trainer, _module: LightningModule) -> None:
        """Restart timing after validation or other work between epochs."""
        if self._active and self._warmup_remaining == 0 and self._window_start is None:
            self._start_window(trainer)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        _module: LightningModule,
        _outputs: Any,
        batch: Any,
        _batch_idx: int,
    ) -> None:
        """Accumulate samples and emit a measurement at the configured cadence."""
        if not self._active:
            return

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            if self._warmup_remaining == 0:
                self._start_window(trainer)
            return

        if self._window_start is None:
            self._start_window(trainer)

        self._window_samples += get_batch_size(batch)
        self._window_batches += 1
        if self._window_batches >= self.log_every_n_steps:
            self._emit(trainer, restart=True)

    def on_train_epoch_end(self, trainer: Trainer, _module: LightningModule) -> None:
        """Flush a partial final window without including validation time."""
        if not self._active:
            return
        if self._window_batches > 0:
            self._emit(trainer, restart=False)
        self._window_start = None

    def _emit(self, trainer: Trainer, restart: bool) -> None:
        """Compute, persist, and optionally print one throughput window."""
        assert self._window_start is not None
        self._synchronize(trainer, "throughput_window_end")
        # Keep every rank on the same control-flow path after the barrier. The
        # lower bound is only defensive for clocks with unusually coarse resolution.
        elapsed = max(time.perf_counter() - self._window_start, 1e-12)

        world_size = trainer.world_size
        device_samples_per_sec = self._window_samples / elapsed
        steps_per_sec = self._window_batches / elapsed
        metrics = {
            "train/throughput/samples_per_sec": device_samples_per_sec * world_size,
            "train/throughput/batches_per_sec": steps_per_sec * world_size,
            "train/throughput/steps_per_sec": steps_per_sec,
            "train/throughput/device_samples_per_sec": device_samples_per_sec,
            "train/throughput/device_batches_per_sec": steps_per_sec,
            "train/throughput/global_batch_size": (
                self._window_samples * world_size / self._window_batches
            ),
            "train/throughput/window_seconds": elapsed,
        }

        if self._writer:
            logger = getattr(trainer, "logger", None)
            if logger is not None:
                logger.log_metrics(metrics, step=trainer.global_step)

            record = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "world_size": world_size,
                "window_batches_per_device": self._window_batches,
                "window_samples_per_device": self._window_samples,
                **metrics,
            }
            assert self.path is not None
            with self.path.open("a") as output:
                output.write(json.dumps(record))
                output.write("\n")

            if self.std_out:
                print(
                    "[throughput] "
                    f"epoch={trainer.current_epoch} step={trainer.global_step} "
                    f"samples/s={metrics['train/throughput/samples_per_sec']:.1f} "
                    f"batches/s={metrics['train/throughput/batches_per_sec']:.3f} "
                    f"steps/s={metrics['train/throughput/steps_per_sec']:.3f} "
                    f"window={elapsed:.2f}s"
                )

        # Keep all ranks aligned and exclude logging/file I/O from the next window.
        self._synchronize(trainer, "throughput_logging_complete")

        self._window_samples = 0
        self._window_batches = 0
        self._window_start = time.perf_counter() if restart else None
