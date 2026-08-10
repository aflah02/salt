import pytest
import torch

from salt.models.pooling import ClassAttentionPooling


def get_inputs(batch_size, seq_len, dim, frac_pad=0.0) -> tuple:
    torch.manual_seed(0)
    x = {"embed_xs": torch.randn(batch_size, seq_len, dim)}
    mask = {"embed_xs": torch.rand(batch_size, seq_len) < frac_pad}
    return x, mask


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("gated", [True, False])
def test_class_attention_pooling(num_layers, gated) -> None:
    batch_size, seq_len, dim = 3, 8, 16
    pool = ClassAttentionPooling(
        input_size=dim,
        num_layers=num_layers,
        dense_kwargs={"activation": "GELU", "gated": gated},
        attn_kwargs={"num_heads": 2},
    )

    x, mask = get_inputs(batch_size, seq_len, dim, frac_pad=0.5)
    out = pool(x, pad_mask=mask)
    assert out.shape == (batch_size, dim)
    assert not torch.isnan(out).any()


def test_class_attention_pooling_no_mask() -> None:
    batch_size, seq_len, dim = 2, 5, 16
    pool = ClassAttentionPooling(input_size=dim, attn_kwargs={"num_heads": 2})
    x, _ = get_inputs(batch_size, seq_len, dim)
    out = pool(x, pad_mask=None)
    assert out.shape == (batch_size, dim)
    assert not torch.isnan(out).any()


def test_class_attention_pooling_fully_padded() -> None:
    # a fully padded jet must stay NaN-free thanks to the appended zero key token
    batch_size, seq_len, dim = 2, 6, 16
    pool = ClassAttentionPooling(input_size=dim, num_layers=2, attn_kwargs={"num_heads": 4})
    x, mask = get_inputs(batch_size, seq_len, dim)
    mask = {"embed_xs": torch.ones(batch_size, seq_len, dtype=torch.bool)}
    out = pool(x, pad_mask=mask)
    assert out.shape == (batch_size, dim)
    assert not torch.isnan(out).any()
