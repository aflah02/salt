"""Shared helpers for the salt unit tests."""

import torch
from torch import nn

from salt.models.attention import Attention


def create_bool_tensor(shape, value):
    return torch.full(shape, value, dtype=torch.bool)


def get_models(dim, num_heads) -> tuple:
    salt_attn = Attention(dim, num_heads=num_heads)
    torch_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    salt_attn.in_proj_weight = torch_attn.in_proj_weight
    salt_attn.in_proj_bias = torch_attn.in_proj_bias
    salt_attn.out_proj.weight = torch_attn.out_proj.weight
    salt_attn.out_proj.bias = torch_attn.out_proj.bias
    return salt_attn, torch_attn


def get_cross_attn_inputs(batch_size, q_len, kv_len, dim, frac_pad=0.0) -> tuple:
    torch.manual_seed(0)
    q = torch.randn(batch_size, q_len, dim)
    kv = torch.randn(batch_size, kv_len, dim)
    kv_mask = torch.rand(batch_size, kv_len) > frac_pad
    kv_mask[:, 0] = False  # Make sure something can send
    return q, kv, kv_mask


def get_self_attn_inputs(batch_size, seq_len, dim, frac_pad=0.0) -> tuple:
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, dim)
    mask = torch.rand(batch_size, seq_len) > frac_pad
    mask[:, 0] = False  # Make sure something can send
    return x, mask
