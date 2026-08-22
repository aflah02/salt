"""Validate the Astral FA3 wheel through Salt using the GN2 attention geometry."""

from importlib.metadata import version
import platform

import torch

from salt.models.attention import Attention
from salt.utils.tensor_utils import redo_padding, undo_padding
import flash_attn_interface

EXPECTED_FA3_VERSION = "3.0.0"


def check_environment() -> None:
    """Check wheel provenance and every GPU visible to the job."""
    import flash_attn_interface

    installed = version("flash-attn-3")
    print("Architecture:", platform.machine())
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("Flash Attention 3:", installed)
    print("FA3 varlen function:", flash_attn_interface.flash_attn_varlen_func)
    print("Visible GPUs:", torch.cuda.device_count())

    assert platform.machine() == "aarch64"
    assert torch.cuda.is_available()
    assert torch.version.cuda is not None and torch.version.cuda.startswith("13.")
    assert installed == EXPECTED_FA3_VERSION, (installed, EXPECTED_FA3_VERSION)
    assert hasattr(flash_attn_interface, "flash_attn_varlen_func")

    for device_index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        print(f"GPU {device_index}: {name}; compute capability={capability}")
        assert capability == (9, 0), (device_index, name, capability)


def check_forward_backward(dtype: torch.dtype) -> None:
    """Run variable-length forward/backward with GN2's D=256, H=8 and S<=41."""
    torch.manual_seed(42)
    batch_size, seq_len, embed_dim, num_heads = 8, 41, 256, 8
    lengths = torch.tensor([41, 40, 33, 24, 16, 8, 2, 1], device="cuda")
    mask = torch.arange(seq_len, device="cuda").unsqueeze(0) >= lengths.unsqueeze(1)
    x = torch.randn(batch_size, seq_len, embed_dim, device="cuda", requires_grad=True)
    attention = Attention(
        embed_dim,
        num_heads=num_heads,
        attn_type="flash3-varlen",
        dropout=0.0,
    ).cuda()

    with torch.autocast("cuda", dtype=dtype):
        packed_x, culens, maxlen = undo_padding(x, mask)
        packed_output = attention(packed_x, culens=culens, maxlen=maxlen)
        output = redo_padding(packed_output, mask)
        loss = output.float().square().mean()

    loss.backward()
    torch.cuda.synchronize()

    assert attention.attn_type == "flash3-varlen"
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    print(
        f"FA3 GN2 forward/backward: SUCCESS; dtype={dtype}; "
        f"tokens={packed_x.shape[0]}; maxlen={maxlen}; loss={loss.item():.8g}"
    )


if __name__ == "__main__":
    check_environment()
    check_forward_backward(torch.float16)
    check_forward_backward(torch.bfloat16)
    print("Flash Attention 3 GN2 preflight: SUCCESS")
