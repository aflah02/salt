from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from salt.models.task import mask_fill_flattened
from salt.modelwrapper import ModelWrapper
from salt.utils.get_structured_input_dict import get_structured_input_dict
from salt.utils.inputs import (
    inputs_sep_with_pad_multi_sequece,
)
from salt.utils.union_find import get_node_assignment_jit

torch.manual_seed(42)


def _compare_aux_outputs(
    outputs_pytorch: dict,
    outputs_onnx: list,
    pad_masks: Sequence[torch.Tensor],
    tasks_to_output: Sequence[str],
) -> None:
    """Compare auxiliary per-track outputs (default Athena export only)."""
    # Test track origin
    if "track_origin" in tasks_to_output:
        pred_pytorch_origin = (
            torch.argmax(outputs_pytorch["tracks"]["track_origin"], dim=-1).detach().numpy()
        )
        onnx_index = tasks_to_output.index("track_origin") - len(tasks_to_output)
        pred_onnx_origin = outputs_onnx[onnx_index]
        assert len(pred_onnx_origin.shape) == 1, (
            "ONNX output for track origin should be a single tensor"
        )
        np.testing.assert_allclose(
            pred_pytorch_origin.squeeze(),
            pred_onnx_origin,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for track origin",
        )

    # Test track vertexing
    if "track_vertexing" in tasks_to_output:
        # Get vertex assignment from PyTorch output
        pred_pytorch_scores = outputs_pytorch["tracks"]["track_vertexing"].detach()
        pred_pytorch_indices = get_node_assignment_jit(pred_pytorch_scores, pad_masks[0])
        pred_pytorch_vtx = mask_fill_flattened(pred_pytorch_indices, pad_masks[0])

        onnx_index = tasks_to_output.index("track_vertexing") - len(tasks_to_output)
        pred_onnx_vtx = outputs_onnx[onnx_index]
        np.testing.assert_allclose(
            pred_pytorch_vtx.squeeze(),
            pred_onnx_vtx,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for vertexing",
        )

    # Test track type
    if "track_type" in tasks_to_output:
        pred_pytorch_type = (
            torch.argmax(outputs_pytorch["tracks"]["track_type"], dim=-1).detach().numpy()
        )
        onnx_index = tasks_to_output.index("track_type") - len(tasks_to_output)
        pred_onnx_type = outputs_onnx[onnx_index]

        assert len(pred_onnx_type.shape) == 1, (
            "ONNX output for track origin should be a single tensor"
        )
        np.testing.assert_allclose(
            pred_pytorch_type.squeeze(),
            pred_onnx_type,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for track type",
        )


def _compare_case(
    pt_model: ModelWrapper,
    session: ort.InferenceSession,
    global_object: str,
    seq_names_salt: Sequence[str],
    seq_names_onnx: Sequence[str],
    variable_map: Mapping[str, list[str]],
    tasks_to_output: Sequence[str],
    *,
    n_batch: int,
    n_seq: int,
    p_valid: float,
    batched: bool,
) -> None:
    """Compare PyTorch vs ONNX outputs for a single synthetic case.

    Handles both the default Athena interface (``batched=False``: one event, no
    padding, sequence inputs without a batch dimension, auxiliary per-track
    outputs) and the standalone batched interface (``batched=True``: an event
    batch dimension, real padding, per-sequence boolean ``*_mask`` inputs,
    global tasks only).
    """
    # Generate synthetic inputs (global + constituent sequences with padding)
    jets, sequences, pad_masks = inputs_sep_with_pad_multi_sequece(
        n_batch,
        [n_seq for _ in seq_names_salt],
        pt_model.input_dims[global_object],
        [pt_model.input_dims[seqn] for seqn in seq_names_salt],
        p_valid=p_valid,
    )

    # Build input dict for PyTorch
    inputs_pytorch = {seqn: seq for seq, seqn in zip(sequences, seq_names_salt, strict=False)}
    inputs_pytorch[global_object] = jets
    if "global" in variable_map:
        inputs_pytorch["global"] = jets.clone()

    masks_pytorch = {seqn: mask for mask, seqn in zip(pad_masks, seq_names_salt, strict=False)}
    structured_input_dict = get_structured_input_dict(inputs_pytorch, variable_map, global_object)

    # Run forward pass of the PyTorch model
    outputs_pytorch = pt_model(inputs_pytorch, masks_pytorch)[0]

    # Collect predictions from PyTorch global tasks
    global_pred_pytorch: list = []
    if global_object in outputs_pytorch:
        global_tasks = [t for t in pt_model.model.tasks if t.input_name == global_object]
        for i, out in enumerate(list(outputs_pytorch[global_object].values())):
            if global_tasks[i].name not in tasks_to_output:
                continue
            onnx_out = global_tasks[i].get_onnx(out, labels=structured_input_dict)
            global_pred_pytorch += [p.detach().numpy() for p in onnx_out]

    # Global input is named "<object>_features" to match the export feature map
    inputs_onnx = {f"{global_object.removesuffix('s')}_features": jets.numpy()}
    for seq, mask, seqn in zip(sequences, pad_masks, seq_names_onnx, strict=False):
        if batched:
            inputs_onnx[seqn] = seq.numpy()
            inputs_onnx[f"{seqn}_mask"] = mask.numpy()
        else:
            inputs_onnx[seqn] = seq.squeeze(0).numpy()

    # Names must match the exported graph, so export-naming drift fails loudly
    expected_names = set(inputs_onnx)
    actual_names = {i.name for i in session.get_inputs()}
    assert expected_names == actual_names, (
        f"ONNX input names {sorted(actual_names)} do not match the names built by the "
        f"validation harness {sorted(expected_names)}"
    )

    # Run ONNX model inference
    outputs_onnx = session.run(None, inputs_onnx)

    # Compare global task predictions
    global_pred_onnx = outputs_onnx[: len(global_pred_pytorch)]
    assert not np.isnan(np.array(global_pred_pytorch)).any()
    assert not np.isnan(np.array(global_pred_onnx)).any()
    assert not (np.array(global_pred_pytorch) == 0).any()
    assert not (np.array(global_pred_onnx) == 0).any()
    np.testing.assert_allclose(
        global_pred_pytorch,
        global_pred_onnx,
        rtol=1e-04,
        atol=1e-04,
        err_msg="Torch vs ONNX check failed for global task",
    )

    # Auxiliary per-track outputs exist only on the default Athena export
    if not batched and n_seq > 0:
        _compare_aux_outputs(outputs_pytorch, outputs_onnx, pad_masks, tasks_to_output)


def compare_outputs(
    pt_model: ModelWrapper,
    onnx_path: str | Path,
    global_object: str,
    seq_names_salt: Sequence[str],
    seq_names_onnx: Sequence[str],
    variable_map: Mapping[str, list[str]],
    tasks_to_output: Sequence[str],
    *,
    batched: bool = False,
) -> None:
    """Validate that PyTorch and ONNX models match across many synthetic cases.

    Parameters
    ----------
    pt_model : ModelWrapper
        PyTorch model used as the reference implementation.
    onnx_path : str | Path
        Path to the exported ONNX model.
    global_object : str
        Name of the global input object, e.g. ``"jets"`` or ``"event"``.
    seq_names_salt : Sequence[str]
        Names of the sequence inputs in the SALT model.
    seq_names_onnx : Sequence[str]
        Names of the corresponding sequence inputs in the ONNX model.
    variable_map : Mapping[str, list[str]]
        Mapping from each input object to its variable names.
    tasks_to_output : Sequence[str]
        Names of the tasks included in the ONNX export.
    batched : bool, optional
        If True, validate the standalone batched interface (event batch
        dimension, per-sequence boolean ``*_mask`` inputs, global tasks only).
        If False, validate the default Athena interface (single event,
        auxiliary per-track outputs). By default False.
    """
    print("\n" + "-" * 100)
    print(f"Validating {'batched standalone ' if batched else ''}ONNX model...")

    # Create ONNX Runtime session with reduced logging
    sess_options = ort.SessionOptions()
    # suppress warnings due to unoptimized subgraphs - https://github.com/microsoft/onnxruntime/issues/14694
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"], sess_options=sess_options
    )

    if batched:
        # Sweep sequence lengths with real padding to exercise the *_mask wiring
        for n_seq in tqdm([1, 8, 40], leave=False):
            for _ in range(3):
                _compare_case(
                    pt_model,
                    session,
                    global_object,
                    seq_names_salt,
                    seq_names_onnx,
                    variable_map,
                    tasks_to_output,
                    n_batch=3,
                    n_seq=n_seq,
                    p_valid=0.8,
                    batched=True,
                )
        print("Success! Pytorch and batched standalone ONNX models are consistent.")
        print("-" * 100)
        return

    # Default Athena interface: loop over sequence lengths (0-39) and trials
    for n_track in tqdm(range(40), leave=False):
        for _ in range(10):
            _compare_case(
                pt_model,
                session,
                global_object,
                seq_names_salt,
                seq_names_onnx,
                variable_map,
                tasks_to_output,
                n_batch=1,
                n_seq=n_track,
                p_valid=1,
                batched=False,
            )

    # Report final success message
    print(
        "Success! Pytorch and ONNX models are consistent, but you should verify this in"
        " Athena.\nFor more info see: https://ftag-salt.docs.cern.ch/export/#athena-validation"
    )
    print("-" * 100)
