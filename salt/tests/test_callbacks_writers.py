"""Unit tests for the writer and MaskFormer callbacks.

Hooks are driven directly with SimpleNamespace fakes and small hand-built
tensors; the full end-to-end writer paths are also covered by the pipeline
tests in test_pipeline.py.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from ftag import get_mock_file
from torch import nn

from salt.callbacks import MaskformerMetrics, PredictionWriter
from salt.callbacks.integrated_gradients_writer import HAS_IG, IntegratedGradientWriter
from salt.callbacks.maskformer_confusion_matrix import (
    MaskformerConfusionMatrix,
    confusion_matrix,
)
from salt.models.task import ClassificationTask


class FakeDataset:
    def __init__(self, filename, num):
        self.filename = filename
        self.num = num
        self.global_object = "jets"
        self.norm_dict = {}
        self.input_map = {"jets": "jets"}
        self.mf_config = None

    def __len__(self):
        return self.num


class LogModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.logged = {}
        self.current_epoch = 0

    def log(self, name, value, **kwargs):
        self.logged[name] = value


def make_jets_task():
    task = ClassificationTask(
        name="jets_classification",
        input_name="jets",
        label="flavour_label",
        class_names=["sig", "bkg"],
        loss=nn.CrossEntropyLoss(),
        dense_config={"input_size": 4, "output_size": 2, "hidden_layers": [4]},
    )
    task.model_name = "salt"
    return task


def make_pw_trainer(tmp_path, ds, batch_size=10):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        datamodule=SimpleNamespace(
            test_dataloader=lambda: SimpleNamespace(dataset=ds),
            batch_size=batch_size,
            test_suff=None,
        ),
        ckpt_path=str(ckpt_dir / "model.ckpt"),
    )


def make_pw_module(task):
    return SimpleNamespace(
        model=SimpleNamespace(tasks=[task], mask_decoder=None),
        global_object="jets",
        name="salt",
    )


# --------------------------- PredictionWriter ---------------------------


def test_prediction_writer_setup_caches_dataset_info(tmp_path):
    ds = FakeDataset(get_mock_file()[0], num=20)
    writer = PredictionWriter()
    trainer = make_pw_trainer(tmp_path, ds)
    writer.setup(trainer, make_pw_module(make_jets_task()), stage="test")
    assert writer.num == 20
    assert writer.batch_size == 10
    assert writer.global_object == "jets"
    assert "jets" in writer._dset_exists
    assert "pt" in writer._dset_dtype_names["jets"]


def test_prediction_writer_setup_ignores_other_stages(tmp_path):
    writer = PredictionWriter()
    writer.setup(SimpleNamespace(), SimpleNamespace(), stage="fit")
    assert not hasattr(writer, "num")


def test_prediction_writer_extra_vars_missing_variable_raises(tmp_path):
    ds = FakeDataset(get_mock_file()[0], num=20)
    writer = PredictionWriter(extra_vars={"jets": ["not_a_variable"]})
    trainer = make_pw_trainer(tmp_path, ds)
    with pytest.raises(ValueError, match="missing"):
        writer.setup(trainer, make_pw_module(make_jets_task()), stage="test")


def test_prediction_writer_extra_vars_unknown_input_type_raises(tmp_path):
    ds = FakeDataset(get_mock_file()[0], num=20)
    writer = PredictionWriter(extra_vars={"weird": ["pt"]})
    trainer = make_pw_trainer(tmp_path, ds)
    with pytest.raises(ValueError, match="not recognized"):
        writer.setup(trainer, make_pw_module(make_jets_task()), stage="test")


def test_prediction_writer_writes_output_h5(tmp_path):
    torch.manual_seed(0)
    task = make_jets_task()
    ds = FakeDataset(get_mock_file()[0], num=20)
    writer = PredictionWriter()
    trainer = make_pw_trainer(tmp_path, ds)
    writer.setup(trainer, make_pw_module(task), stage="test")

    for batch_idx in range(2):
        outputs = {"jets": {"jets_classification": torch.randn(10, 2)}}
        batch = (None, {}, {})
        writer.on_test_batch_end(trainer, None, outputs, batch, batch_idx)
    writer.on_test_end(trainer, None)

    assert writer.output_path.exists()
    with h5py.File(writer.output_path) as f:
        jets = f["jets"]
        assert len(jets) == 20
        assert "salt_psig" in jets.dtype.names
        assert "salt_pbkg" in jets.dtype.names
        assert "pt" in jets.dtype.names
        probs = jets["salt_psig"][:] + jets["salt_pbkg"][:]
        np.testing.assert_allclose(probs, np.ones(20), rtol=1e-5)


# --------------------------- MaskformerMetrics ---------------------------


def make_mf_trainer(class_names):
    mf_config = SimpleNamespace(object=SimpleNamespace(class_names=class_names))
    return SimpleNamespace(
        fast_dev_run=False,
        device_ids=[0],
        datamodule=SimpleNamespace(
            train_dataloader=lambda: SimpleNamespace(dataset=SimpleNamespace(mf_config=mf_config))
        ),
    )


def test_maskformer_metrics_requires_mask_decoder():
    cb = MaskformerMetrics()
    module = LogModule()
    module.model = SimpleNamespace(mask_decoder=None)
    with pytest.raises(ValueError, match="mask_decoder"):
        cb.setup(make_mf_trainer(["vertex", "null"]), module, stage="fit")


def test_maskformer_metrics_logs_validation_metrics():
    torch.manual_seed(0)
    cb = MaskformerMetrics()
    module = LogModule()
    module.model = SimpleNamespace(mask_decoder=True, tasks=[])
    trainer = make_mf_trainer(["vertex", "null"])
    cb.setup(trainer, module, stage="fit")

    b, n_obj, n_trk = 2, 3, 5
    outputs = {
        "outputs": {
            "preds": {
                "objects": {
                    "class_logits": torch.randn(b, n_obj, 2),
                    "masks": torch.randn(b, n_obj, n_trk),
                }
            },
            "labels": {
                "objects": {
                    "object_class": torch.randint(0, 2, (b, n_obj)),
                    "masks": torch.rand(b, n_obj, n_trk) > 0.5,
                }
            },
            "pad_masks": {"tracks": torch.zeros(b, n_trk, dtype=torch.bool)},
        }
    }
    cb.on_validation_batch_end(trainer, module, outputs, None, 0)

    for key in (
        "val/class_exact_match",
        "val/class_accuracy_micro",
        "val/notnull_eff",
        "val/notnull_pur",
        "val/vertex_eff",
        "val/vertex_pur",
        "val/query_perfect_match_eff",
        "val/query_loose_match_fake",
    ):
        assert key in module.logged


# --------------------------- MaskformerConfusionMatrix ---------------------------


def test_confusion_matrix_function():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=range(3))
    expected = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]])
    np.testing.assert_array_equal(cm, expected)


def test_confusion_matrix_function_infers_labels():
    cm = confusion_matrix([0, 5], [5, 5])
    assert cm.shape == (2, 2)
    assert cm.sum() == 2


def test_maskformer_confusion_matrix_requires_mask_decoder():
    cb = MaskformerConfusionMatrix()
    module = LogModule()
    module.model = SimpleNamespace(mask_decoder=None)
    with pytest.raises(ValueError, match="mask_decoder"):
        cb.setup(make_mf_trainer(["vertex", "null"]), module, stage="fit")


def test_maskformer_confusion_matrix_accumulates_and_resets():
    cb = MaskformerConfusionMatrix()
    module = LogModule()
    module.model = SimpleNamespace(mask_decoder=True)
    module.logger = SimpleNamespace()  # no .experiment -> nothing logged
    trainer = make_mf_trainer(["vertex", "null"])
    cb.setup(trainer, module, stage="fit")

    outputs = {
        "outputs": {
            "preds": {"objects": {"class_logits": torch.randn(2, 3, 2)}},
            "labels": {"objects": {"object_class": torch.randint(0, 2, (2, 3))}},
        }
    }
    cb.on_validation_batch_end(trainer, module, outputs, None, 0)
    assert len(cb.val_preds) == 1
    assert cb.val_preds[0].shape == (6,)

    cb.on_validation_epoch_end(trainer, module)
    assert cb.val_preds == []
    assert cb.val_targets == []


def test_maskformer_confusion_matrix_off_cadence_clears_without_logging():
    cb = MaskformerConfusionMatrix(log_every_n_epochs=2)
    module = LogModule()
    module.current_epoch = 1
    trainer = SimpleNamespace(fast_dev_run=False)
    cb.val_preds = [torch.tensor([0, 1])]
    cb.val_targets = [torch.tensor([0, 1])]
    cb.on_validation_epoch_end(trainer, module)
    assert cb.val_preds == []


# --------------------------- IntegratedGradientWriter ---------------------------


@pytest.mark.skipif(HAS_IG, reason="captum/salt-attribution installed")
def test_integrated_gradients_writer_requires_optional_deps():
    with pytest.raises(ImportError, match="captum"):
        IntegratedGradientWriter(input_keys={}, output_keys=[])
