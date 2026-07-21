import pytest
import torch

from salt.utils.tensor_utils import (
    add_dims,
    attach_context,
    attach_context_single,
    flatten_tensor_dict,
    masked_softmax,
    maybe_flatten_tensors,
    redo_padding,
    undo_padding,
)


@pytest.fixture
def tensor_dict():
    torch.manual_seed(0)
    return {
        "a": torch.randn(2, 3, 4),
        "b": torch.randn(2, 5, 4),
        "c": torch.randn(2, 2, 4),
    }


def test_maybe_flatten_tensors_passthrough():
    x = torch.randn(2, 3)
    assert maybe_flatten_tensors(x) is x


def test_maybe_flatten_tensors_dict(tensor_dict):
    out = maybe_flatten_tensors(tensor_dict)
    assert out.shape == (2, 10, 4)


def test_flatten_tensor_dict_default(tensor_dict):
    out = flatten_tensor_dict(tensor_dict)
    expected = torch.cat([tensor_dict["a"], tensor_dict["b"], tensor_dict["c"]], dim=1)
    torch.testing.assert_close(out, expected)


def test_flatten_tensor_dict_include(tensor_dict):
    out = flatten_tensor_dict(tensor_dict, include=["a", "c"])
    expected = torch.cat([tensor_dict["a"], tensor_dict["c"]], dim=1)
    torch.testing.assert_close(out, expected)


def test_flatten_tensor_dict_exclude(tensor_dict):
    out = flatten_tensor_dict(tensor_dict, exclude=["b"])
    expected = torch.cat([tensor_dict["a"], tensor_dict["c"]], dim=1)
    torch.testing.assert_close(out, expected)


def test_flatten_tensor_dict_include_and_exclude_raises(tensor_dict):
    with pytest.raises(ValueError, match="together"):
        flatten_tensor_dict(tensor_dict, include=["a"], exclude=["b"])


def test_masked_softmax_no_mask_matches_softmax():
    torch.manual_seed(0)
    x = torch.randn(2, 5)
    torch.testing.assert_close(masked_softmax(x, None), torch.softmax(x, dim=-1))


def test_masked_softmax_masked_positions_are_zero():
    torch.manual_seed(0)
    x = torch.randn(2, 5)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, -2:] = True
    out = masked_softmax(x, mask)
    assert torch.all(out[mask] == 0)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(2))


def test_undo_redo_padding_round_trip():
    torch.manual_seed(0)
    x = torch.randn(3, 6, 4)
    mask = torch.rand(3, 6) > 0.5
    mask[:, 0] = False  # keep at least one valid element per sequence

    unpadded, culens, maxlen = undo_padding(x, mask)
    assert unpadded.shape == (int((~mask).sum()), 4)
    assert culens.dtype == torch.int32
    assert culens.shape == (4,)
    assert culens[0] == 0
    assert maxlen == int((~mask).sum(dim=-1).max())

    restored = redo_padding(unpadded, mask)
    torch.testing.assert_close(restored, x * ~mask.unsqueeze(-1))


def test_add_dims():
    x = torch.randn(2, 5)
    out = add_dims(x, 4)
    assert out.shape == (2, 1, 1, 5)
    assert add_dims(x, 2).shape == x.shape


def test_add_dims_raises_on_smaller_ndim():
    x = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="smaller"):
        add_dims(x, 2)


def test_attach_context_single_same_rank():
    x = torch.randn(2, 4)
    context = torch.randn(2, 3)
    out = attach_context_single(x, context)
    assert out.shape == (2, 7)
    torch.testing.assert_close(out[:, :3], context)


def test_attach_context_single_broadcast():
    x = torch.randn(2, 5, 4)
    context = torch.randn(2, 3)
    out = attach_context_single(x, context)
    assert out.shape == (2, 5, 7)
    torch.testing.assert_close(out[:, 2, :3], context)


def test_attach_context_single_missing_context_raises():
    with pytest.raises(RuntimeError, match="missing"):
        attach_context_single(torch.randn(2, 4), None)


def test_attach_context_single_overranked_context_raises():
    with pytest.raises(ValueError, match="more dimensions"):
        attach_context_single(torch.randn(2, 4), torch.randn(2, 5, 3))


def test_attach_context_dict():
    x = {"a": torch.randn(2, 5, 4), "b": torch.randn(2, 3, 4)}
    context = torch.randn(2, 3)
    out = attach_context(x, context)
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (2, 5, 7)
    assert out["b"].shape == (2, 3, 7)
