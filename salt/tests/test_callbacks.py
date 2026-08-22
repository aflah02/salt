"""Unit tests for the lightweight Lightning callbacks.

The callbacks are exercised by calling their hooks directly with SimpleNamespace
fakes instead of a real Trainer. The full save/write paths of SaveConfigCallback,
PredictionWriter and the writer callbacks are covered end-to-end by the pipeline
tests in test_pipeline.py.
"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from salt.callbacks import (
    Checkpoint,
    ConfusionMatrixCallback,
    GradientLoggerCallback,
    PerformanceWriter,
    SaveConfigCallback,
    ThroughputLogger,
    WeightLoggerCallback,
)
from salt.callbacks.throughput import get_batch_size
from salt.callbacks.saveconfig import get_attr


def make_fake_trainer(**attrs):
    defaults = {
        "fast_dev_run": False,
        "device_ids": [0],
        "is_global_zero": True,
        "strategy": SimpleNamespace(broadcast=lambda x: x),
    }
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.logged = {}
        self.current_epoch = 0

    def forward(self, x):
        return self.linear(x)

    def log(self, name, value, **kwargs):
        self.logged[name] = value


# --------------------------- Checkpoint ---------------------------


def test_checkpoint_setup_sets_dirpath(tmp_path):
    ckpt = Checkpoint(monitor_loss="val_loss")
    trainer = make_fake_trainer(log_dir=str(tmp_path))
    ckpt.setup(trainer, TinyModule(), stage="fit")
    assert ckpt.dirpath == str(tmp_path / "ckpts")
    assert "val_loss" in ckpt.filename


def test_checkpoint_setup_fast_dev_run_leaves_dirpath(tmp_path):
    ckpt = Checkpoint(monitor_loss="val_loss")
    trainer = make_fake_trainer(log_dir=str(tmp_path), fast_dev_run=True)
    ckpt.setup(trainer, TinyModule(), stage="fit")
    assert ckpt.dirpath is None


def test_checkpoint_setup_malformed_s3_raises(tmp_path):
    ckpt = Checkpoint(monitor_loss="val_loss")
    trainer = make_fake_trainer(log_dir=str(tmp_path) + "/s3:/bucket")
    with pytest.raises(ValueError, match="should start with"):
        ckpt.setup(trainer, TinyModule(), stage="fit")


# --------------------------- Gradient/weight loggers ---------------------------


def test_gradient_logger_logs_at_cadence():
    cb = GradientLoggerCallback(log_every_n_steps=10)
    module = TinyModule()
    trainer = make_fake_trainer(global_step=0)
    cb.setup(trainer, module, stage="fit")

    loss = module(torch.randn(3, 2)).sum()
    loss.backward()
    cb.on_after_backward(trainer, module)
    assert "gradients_grad/linear.weight_mean" in module.logged
    assert "gradients_grad/linear.bias_std" in module.logged
    assert "gradient_total_magnitude" in module.logged
    assert "gradient_average_magnitude" in module.logged


def test_gradient_logger_skips_off_cadence():
    cb = GradientLoggerCallback(log_every_n_steps=10)
    module = TinyModule()
    trainer = make_fake_trainer(global_step=3)
    cb.setup(trainer, module, stage="fit")
    cb.on_after_backward(trainer, module)
    assert module.logged == {}


def test_weight_logger_logs_weight_stats():
    cb = WeightLoggerCallback(log_every_n_steps=10)
    module = TinyModule()
    trainer = make_fake_trainer(global_step=0)
    cb.setup(trainer, module, stage="fit")
    cb.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
    assert "weights_weights/linear.weight_mean" in module.logged
    assert "weights_weights/linear.bias_std" in module.logged


# --------------------------- PerformanceWriter ---------------------------


def test_performance_writer(tmp_path):
    cb = PerformanceWriter(dir_path=str(tmp_path), add_metrics=["extra_metric"])
    module = TinyModule()
    module.current_epoch = 2
    trainer = make_fake_trainer(
        log_dir=str(tmp_path),
        state=SimpleNamespace(stage="validate"),
        callback_metrics={"val_loss": torch.tensor(0.5), "extra_metric": torch.tensor(1.5)},
    )
    cb.setup(trainer, module, stage="fit")
    assert cb.path.exists()

    cb.on_validation_epoch_end(trainer, module)
    record = json.loads(cb.path.read_text().splitlines()[0])
    assert record["epoch"] == 2
    assert record["val_loss"] == "0.50000"
    assert record["extra_metric"] == "1.50000"


def test_performance_writer_skips_outside_validate(tmp_path):
    cb = PerformanceWriter(dir_path=str(tmp_path))
    trainer = make_fake_trainer(
        log_dir=str(tmp_path), state=SimpleNamespace(stage="sanity_check")
    )
    cb.setup(trainer, TinyModule(), stage="fit")
    cb.on_validation_epoch_end(trainer, TinyModule())
    assert cb.path.read_text() == ""


# --------------------------- ThroughputLogger ---------------------------


def test_get_batch_size_from_salt_batch():
    batch = (
        {
            "jets": torch.zeros(3, 2),
            "tracks": torch.zeros(3, 40, 19),
        },
        {"tracks": torch.zeros(3, 40, dtype=torch.bool)},
        {"jets": {"flavour_label": torch.zeros(3, dtype=torch.long)}},
    )
    assert get_batch_size(batch) == 3


def test_throughput_logger_writes_global_rates(tmp_path):
    class FakeLogger:
        def __init__(self):
            self.metrics = []

        def log_metrics(self, metrics, step):
            self.metrics.append((metrics, step))

    logger = FakeLogger()
    trainer = make_fake_trainer(
        log_dir=str(tmp_path),
        logger=logger,
        world_size=4,
        global_step=0,
        current_epoch=0,
        strategy=SimpleNamespace(root_device=torch.device("cpu"), barrier=lambda _name: None),
    )
    callback = ThroughputLogger(log_every_n_steps=2, warmup_steps=0, std_out=False)
    callback.setup(trainer, TinyModule(), stage="fit")
    callback.on_train_start(trainer, TinyModule())

    batch = ({"jets": torch.zeros(3, 2)}, {}, {})
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, TinyModule(), None, batch, 0)
    trainer.global_step = 2
    callback.on_train_batch_end(trainer, TinyModule(), None, batch, 1)

    records = callback.path.read_text().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["world_size"] == 4
    assert record["window_batches_per_device"] == 2
    assert record["window_samples_per_device"] == 6
    assert record["train/throughput/global_batch_size"] == 12
    assert record["train/throughput/samples_per_sec"] > 0
    assert record["train/throughput/batches_per_sec"] > 0
    assert record["train/throughput/steps_per_sec"] > 0
    assert len(logger.metrics) == 1
    assert logger.metrics[0][1] == 2


# --------------------------- ConfusionMatrixCallback ---------------------------


def make_cm_trainer():
    task = SimpleNamespace(
        name="jets_classification",
        input_name="jets",
        label="flavour_label",
        class_names=["b", "c", "u"],
    )
    return make_fake_trainer(model=SimpleNamespace(model=SimpleNamespace(tasks=[task])))


def test_confusion_matrix_setup_resolves_task():
    cb = ConfusionMatrixCallback(task_name="jets_classification")
    cb.setup(make_cm_trainer(), None, stage="fit")
    assert cb.task_input_name == "jets"
    assert cb.task_label_name == "flavour_label"
    assert cb.task_class_names == ["b", "c", "u"]


def test_confusion_matrix_setup_class_name_override():
    cb = ConfusionMatrixCallback(
        task_name="jets_classification", class_names_override={"b": "bjets"}
    )
    cb.setup(make_cm_trainer(), None, stage="fit")
    assert cb.task_class_names == ["bjets", "c", "u"]


def test_confusion_matrix_accumulates_and_resets():
    cb = ConfusionMatrixCallback(task_name="jets_classification")
    trainer = make_cm_trainer()
    cb.setup(trainer, None, stage="fit")

    preds = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.1, 0.9]])
    outputs = {
        "outputs": {
            "preds": {"jets": {"jets_classification": preds}},
            "labels": {"jets": {"flavour_label": torch.tensor([0, 1])}},
        }
    }
    cb.on_validation_batch_end(trainer, None, outputs, None, 0)
    assert [int(x) for x in cb.pred_labels] == [0, 2]
    assert [int(x) for x in cb.truth_labels] == [0, 1]

    cb.on_validation_epoch_end(make_fake_trainer(logger=None, current_epoch=0), None)
    assert cb.pred_labels == []
    assert cb.truth_labels == []


# --------------------------- SaveConfigCallback ---------------------------


def test_saveconfig_refuses_to_overwrite(tmp_path):
    (tmp_path / "config.yaml").touch()
    cb = SaveConfigCallback(parser=None, config=None)
    trainer = make_fake_trainer(
        log_dir=str(tmp_path),
        strategy=SimpleNamespace(broadcast=lambda x: x),
    )
    with pytest.raises(RuntimeError, match="to NOT exist"):
        cb.setup(trainer, TinyModule(), stage="fit")


def test_saveconfig_get_attr(tmp_path):
    fname = tmp_path / "test.h5"
    with h5py.File(fname, "w") as f:
        f.attrs["n_jets"] = np.int64(100)
        f.attrs["config"] = json.dumps({"a": 1})
        f.attrs["plain"] = "hello"
        f.create_dataset("jets", data=np.zeros(1)).attrs["local"] = "yes"

    with h5py.File(fname) as f:
        assert get_attr(f, "n_jets") == 100
        assert isinstance(get_attr(f, "n_jets"), int)
        assert get_attr(f, "config") == {"a": 1}
        assert get_attr(f, "plain") == "hello"
        assert get_attr(f, "local", key="jets") == "yes"
        assert get_attr(f, "not_there") is None
