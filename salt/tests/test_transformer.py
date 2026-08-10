import pytest
import torch
from torch import nn

from salt.models.attention import Attention
from salt.models.layernorm import RMSNorm
from salt.models.transformer import (
    DecoderLayer,
    EncoderLayer,
    NormResidual,
    Transformer,
    change_attn_backends,
)
from salt.tests.helpers import get_cross_attn_inputs, get_self_attn_inputs

N_BATCH = 10
Q_SEQ = 20
DIM = 16


@pytest.mark.parametrize("ls_init", [None, 0.1])
@pytest.mark.parametrize("drop_path", [0.0, 0.9])
@pytest.mark.parametrize("norm_type", ["pre", "post", "none"])
def test_norm_residual(ls_init, drop_path, norm_type):
    x = torch.randn(N_BATCH, Q_SEQ, DIM)
    block = nn.Linear(DIM, DIM)
    nr = NormResidual(
        block, ls_init=ls_init, drop_path=drop_path, embed_dim=DIM, norm_type=norm_type
    )
    y = nr(x)
    assert y.shape == x.shape
    assert not y.isnan().any()


@pytest.mark.parametrize("norm_type", ["hybrid", "none"])
def test_encoder_layer(norm_type):
    x, mask = get_self_attn_inputs(5, 10, 32, 0.5)
    encoder = EncoderLayer(
        embed_dim=32,
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        norm_type=norm_type,
    )
    x = encoder(x, mask=mask)
    assert x.shape == (5, 10, 32)
    assert not x.isnan().any()


def test_encoder_layer_mup():
    x, mask = get_self_attn_inputs(5, 10, 32, 0.5)
    encoder = EncoderLayer(
        embed_dim=32,
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        mup=True,
    )
    x = encoder(x, mask=mask)
    assert x.shape == (5, 10, 32)
    assert not x.isnan().any()


def test_decoder_layer():
    q, kv, kv_mask = get_cross_attn_inputs(5, 10, 5, 32, 0.5)
    decoder = DecoderLayer(
        embed_dim=32,
        dense_kwargs={"activation": "SiLU", "gated": True},
        attn_kwargs={"num_heads": 2},
    )
    x = decoder(q, kv=kv, kv_mask=kv_mask)
    assert x.shape == q.shape
    assert not x.isnan().any()


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("out_dim", [None, 64])
@pytest.mark.parametrize("num_registers", [1, 4])
@pytest.mark.parametrize("drop_registers", [False, True])
def test_transformer_tensor_input(num_registers, out_dim, num_layers, drop_registers):
    x, mask = get_self_attn_inputs(5, 10, 32, 0.5)
    out_dim_expected = out_dim or 32
    seq_len_expected = 10 + num_registers if not drop_registers else 10
    trans = Transformer(
        num_layers=num_layers,
        embed_dim=32,
        out_dim=out_dim,
        attn_type="torch-math",
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        num_registers=num_registers,
        drop_registers=drop_registers,
    )
    x, mask = trans(x, pad_mask=mask)
    assert x.shape == (5, seq_len_expected, out_dim_expected)
    assert not x.isnan().any()


@pytest.mark.parametrize("num_registers", [1, 4])
def test_transformer_dict_input(num_registers):
    x1, m1 = get_self_attn_inputs(5, 10, 32, 0.5)
    x2, m2 = get_self_attn_inputs(5, 3, 32, 0.5)
    x3, m3 = get_self_attn_inputs(5, 2, 32, 0.5)
    x = {"m1": x1, "m2": x2, "m3": x3}  # Multimodal inputs
    mask = {"m1": m1, "m2": m2, "m3": m3}
    trans = Transformer(
        num_layers=3,
        embed_dim=32,
        attn_type="torch-math",
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        num_registers=num_registers,
    )
    x, mask = trans(x, pad_mask=mask)
    assert x.shape == (5, 10 + 3 + 2 + num_registers, 32)
    assert all(k in mask for k in ["m1", "m2", "m3", "REGISTERS"])


@pytest.mark.parametrize("schedule", ["uniform", "linear"])
def test_transformer_drop_path_schedule(schedule):
    num_layers = 4
    drop_path = 0.2
    trans = Transformer(
        num_layers=num_layers,
        embed_dim=32,
        attn_type="torch-math",
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
        drop_path=drop_path,
        drop_path_schedule=schedule,
    )
    rates = [layer.attn.drop_path.drop_prob for layer in trans.layers]
    if schedule == "uniform":
        assert all(r == drop_path for r in rates)
    else:
        expected = [drop_path * 0.5 * (1 + d / (num_layers - 1)) for d in range(num_layers)]
        assert rates == pytest.approx(expected)

    # forward pass still runs and is NaN-free
    x, mask = get_self_attn_inputs(5, 10, 32, 0.5)
    out, _ = trans(x, pad_mask=mask)
    assert not out.isnan().any()


def test_RMSNorm():
    rmsnorm = RMSNorm(10)
    x = torch.randn(5, 10)
    rmsnorm(x)


def test_change_attn_backends():
    model = Transformer(
        num_layers=3,
        embed_dim=32,
        attn_type="torch-math",
        dense_kwargs={"activation": "SiLU"},
        attn_kwargs={"num_heads": 2},
    )

    # change the backend
    change_attn_backends(model, "torch-meff")
    assert model.attn_type == "torch-meff"
    for layer in model.layers:
        assert layer.attn.fn.attn_type == "torch-meff"

    # no cuda, so it should not be able to set flash-varlen, and isntead fall back to torch-math
    if not torch.cuda.is_available():
        with pytest.warns(UserWarning):
            change_attn_backends(model, "flash-varlen")
        assert model.attn_type == "torch-math"
        for layer in model.layers:
            assert layer.attn.fn.attn_type == "torch-math"

    # check this works for a module that wraps a transformer
    wrapper = nn.Sequential(model)
    change_attn_backends(wrapper, "torch-flash")
    assert model.attn_type == "torch-flash"
    for layer in model.layers:
        assert layer.attn.fn.attn_type == "torch-flash"

    # check this works for a base attention layer
    attn = Attention(32, num_heads=2, attn_type="torch-math")
    change_attn_backends(attn, "torch-flash")
    assert attn.attn_type == "torch-flash"
