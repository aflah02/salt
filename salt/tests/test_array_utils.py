import numpy as np
import pytest
import torch

from salt.utils.array_utils import join_structured_arrays, listify, maybe_copy, maybe_pad
from salt.utils.edge_features import calculate_edge_features, check_edge_config
from salt.utils.get_structured_input_dict import get_structured_input_dict


def test_join_structured_arrays():
    a = np.zeros(3, dtype=[("x", "f4"), ("y", "f4")])
    b = np.ones(3, dtype=[("z", "i4")])
    out = join_structured_arrays([a, b])
    assert out.dtype.names == ("x", "y", "z")
    assert out.shape == (3,)
    np.testing.assert_array_equal(out["z"], b["z"])


def test_listify():
    assert listify(None) is None
    assert listify(1) == [1]
    assert listify([1, 2]) == [1, 2]


def test_maybe_pad():
    src = np.ones((2, 3))
    tgt = np.zeros((2, 5))
    out = maybe_pad(src, tgt)
    assert out.shape == (2, 5)
    np.testing.assert_array_equal(out[:, 3:], 0)
    assert maybe_pad(src, np.zeros((2, 3))) is src


def test_maybe_copy():
    contiguous = np.ones((3, 3))
    assert maybe_copy(contiguous) is contiguous
    non_contiguous = np.ones((3, 3))[:, ::2]
    out = maybe_copy(non_contiguous)
    assert out is not non_contiguous
    assert out.flags.c_contiguous


def test_get_structured_input_dict():
    inputs = {"jets": np.arange(6, dtype="f4").reshape(3, 2)}
    variable_map = {"jets": ["pt", "eta"]}
    out = get_structured_input_dict(inputs, variable_map, "jets")
    assert set(out) == {"jets"}
    np.testing.assert_array_equal(out["jets"]["pt"], inputs["jets"][:, 0])
    np.testing.assert_array_equal(out["jets"]["eta"], inputs["jets"][:, 1])


def test_check_edge_config_valid():
    check_edge_config(["dR", "z", "kt", "isSelfLoop"], ["eta", "phi", "pt"])


def test_check_edge_config_unknown_feature():
    with pytest.raises(ValueError, match="not recognized"):
        check_edge_config(["not_a_feature"], ["eta", "phi", "pt"])


def test_check_edge_config_missing_variable():
    with pytest.raises(ValueError, match="required for edge features"):
        check_edge_config(["dR"], ["pt"])


def test_calculate_edge_features_shape_and_finite():
    torch.manual_seed(0)
    batch = torch.rand(2, 4, 4) + 0.1
    indices_map = {"pt": 0, "eta": 1, "phi": 2, "energy": 3}
    variables = ["dR", "kt", "z", "isSelfLoop", "mass"]
    out = calculate_edge_features(batch, indices_map, variables)
    assert out.shape == (2, 4, 4, len(variables))
    assert torch.isfinite(out).all()


def test_calculate_edge_features_self_loop_is_identity():
    batch = torch.rand(2, 4, 3)
    out = calculate_edge_features(batch, {"pt": 0, "eta": 1, "phi": 2}, ["isSelfLoop"])
    expected = torch.eye(4).expand(2, -1, -1)
    torch.testing.assert_close(out[..., 0], expected)
