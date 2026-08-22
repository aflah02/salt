import importlib.util

import pytest
import torch
from torch import nn

from salt.models.attention import Attention, merge_masks, projection_packed
from salt.models.transformer import (
    redo_padding,
    undo_padding,
)
from salt.tests.helpers import (
    create_bool_tensor,
    get_cross_attn_inputs,
    get_models,
    get_self_attn_inputs,
)

N_BATCH = 10
Q_SEQ = 20
KV_SEQ = 10
DIM = 16


class TestMergeMasks:
    def test_merge_masks_none_inputs(self):
        q_shape = (N_BATCH, Q_SEQ, DIM)
        mask = merge_masks(None, None, q_shape)
        assert mask is None

    def test_merge_masks_only_attn_mask(self):
        q_shape = (N_BATCH, Q_SEQ, DIM)
        attn_shape = (N_BATCH, Q_SEQ, KV_SEQ)
        attn_mask = create_bool_tensor(attn_shape, False)
        mask = merge_masks(None, attn_mask, q_shape)
        assert mask.shape == (N_BATCH, 1, Q_SEQ, KV_SEQ)

    def test_merge_masks_only_kv_mask(self):
        q_shape = (N_BATCH, Q_SEQ, DIM)
        k_shape = (N_BATCH, KV_SEQ, DIM)
        kv_mask = create_bool_tensor(k_shape[:-1], False)
        mask = merge_masks(kv_mask, None, q_shape)
        assert mask.shape == (N_BATCH, 1, Q_SEQ, KV_SEQ)

    def test_merge_masks_attn_and_kv_masks(self):
        q_shape = (N_BATCH, Q_SEQ, DIM)
        k_shape = (N_BATCH, KV_SEQ, DIM)
        attn_shape = (N_BATCH, Q_SEQ, KV_SEQ)
        kv_mask = create_bool_tensor(k_shape[:-1], False)
        attn_mask = create_bool_tensor(attn_shape, True)
        mask = merge_masks(kv_mask, attn_mask, q_shape)
        assert mask.shape == (N_BATCH, 1, Q_SEQ, KV_SEQ)
        assert torch.all(mask)


def test_padding_mask():
    torch_attn = nn.MultiheadAttention(8, 1, batch_first=True)

    # this is a correct full attention mask for padded inputs
    # we only need to pad the keys so that no query receives information from them
    # if we were to also pad the queries, the padded tokens would become nan as
    # they would not recieve any incomming signal
    attn_mask = torch.tensor([[[False, False, True], [False, False, True], [False, False, True]]])

    # confirm that the value of the padded tokens has no effect on the output
    x = torch.ones(1, 3, 8)
    x[:, 2] = 0
    out1 = torch_attn(x, x, x, attn_mask=attn_mask)[0]
    x[:, 2] = 10
    out2 = torch_attn(x, x, x, attn_mask=attn_mask)[0]
    torch.testing.assert_close(out1, out2)


def test_mup_initialization():
    dim = 32
    num_heads = 4
    attn = Attention(dim, num_heads=num_heads, mup=True)
    x = torch.randn(2, 10, dim)

    # Check that q projection is initalized to zero
    q, _, _ = projection_packed(
        x,
        x,
        attn.in_proj_weight,
        attn.in_proj_bias,
    )
    assert torch.allclose(q, 0 * q)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("q_len", [1, 10])
@pytest.mark.parametrize("kv_len", [1, 10])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5, 0.9])
def test_cross_attention(
    batch_size,
    q_len,
    kv_len,
    dim,
    frac_pad,
) -> None:
    salt_attn, torch_attn = get_models(dim, 2)
    q, kv, kv_mask = get_cross_attn_inputs(batch_size, q_len, kv_len, dim, frac_pad)
    custom_output = salt_attn(q, kv, kv_mask=kv_mask)
    torch_output, _ = torch_attn(q, kv, kv, key_padding_mask=kv_mask)
    torch.testing.assert_close(custom_output, torch_output)
    assert not torch.isnan(custom_output).any()


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("seq_len", [1, 2, 10])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5, 0.9])
def test_self_attention(
    batch_size,
    seq_len,
    dim,
    num_heads,
    frac_pad,
) -> None:
    salt_attn, torch_attn = get_models(dim, num_heads)
    x, mask = get_self_attn_inputs(batch_size, seq_len, dim, frac_pad)
    custom_output = salt_attn(x, mask=mask)
    torch_output, _ = torch_attn(x, x, x, key_padding_mask=mask)
    torch.testing.assert_close(custom_output, torch_output)
    assert not torch.isnan(custom_output).any()


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("seq_len", [1, 2, 10])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("frac_pad", [0.0, 0.5])
@pytest.mark.parametrize("do_qk_norm", [True, False])
@pytest.mark.parametrize("do_v_norm", [True, False])
@pytest.mark.parametrize(
    "attn_type", ["torch-flash", "torch-meff", "flash-varlen", "flash3-varlen"]
)
@pytest.mark.gpu
def test_attention_backends(
    batch_size,
    seq_len,
    dim,
    num_heads,
    frac_pad,
    do_qk_norm,
    do_v_norm,
    attn_type,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if attn_type == "flash-varlen" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn not available")
    if attn_type == "flash3-varlen":
        if importlib.util.find_spec("flash_attn_3") is None:
            pytest.skip("flash_attn_3 not available")
        if torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("flash_attn_3 wheel requires an SM90 Hopper GPU")

    # FlashVarlenAttention requires half precision
    with torch.autocast("cuda", enabled=True):
        # Get the inputs and move to device
        x, mask = get_self_attn_inputs(batch_size, seq_len, dim, frac_pad)
        x = x.cuda()
        mask = mask.cuda()

        attn = Attention(dim, num_heads=num_heads, do_qk_norm=do_qk_norm, do_v_norm=do_v_norm).to(
            "cuda"
        )
        # Change the masking to None for the torch backends as they dont support it
        if "torch" in attn_type:
            mask = None

        # Perform the standard attention (math)
        output = attn(x, mask=mask)

        # ensure zero padded
        if mask is not None:
            output *= ~mask.unsqueeze(-1)

        # Switch to the attention backend
        attn.set_backend(attn_type)
        if attn_type in {"flash-varlen", "flash3-varlen"}:
            x_p, culens, maxlen = undo_padding(x, mask)
            output_2 = attn(x_p, mask=mask, culens=culens, maxlen=maxlen)
            output_2 = redo_padding(output_2, mask)
        else:
            output_2 = attn(x, mask=mask)

        # Test all close with less strict due to half precision
        tolerance = 3e-2 if attn_type == "flash3-varlen" else 1e-3
        torch.testing.assert_close(output, output_2, atol=tolerance, rtol=tolerance)
        assert not torch.isnan(output_2).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.gpu
def test_flash3_varlen_gn2_forward_backward(dtype) -> None:
    """Exercise FA3 with the GN2 attention dimensions and variable sequence lengths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if importlib.util.find_spec("flash_attn_3") is None:
        pytest.skip("flash_attn_3 not available")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("flash_attn_3 wheel requires an SM90 Hopper GPU")

    torch.manual_seed(42)
    batch_size, seq_len, dim, num_heads = 8, 41, 256, 8
    lengths = torch.tensor([41, 40, 33, 24, 16, 8, 2, 1], device="cuda")
    mask = torch.arange(seq_len, device="cuda").unsqueeze(0) >= lengths.unsqueeze(1)
    x = torch.randn(batch_size, seq_len, dim, device="cuda", requires_grad=True)
    attn = Attention(dim, num_heads=num_heads, attn_type="flash3-varlen").cuda()

    with torch.autocast("cuda", dtype=dtype):
        x_p, culens, maxlen = undo_padding(x, mask)
        output = attn(x_p, culens=culens, maxlen=maxlen)
        output = redo_padding(output, mask)
        loss = output.float().square().mean()

    loss.backward()

    assert output.shape == (batch_size, seq_len, dim)
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in attn.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
