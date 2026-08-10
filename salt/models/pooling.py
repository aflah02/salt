import torch
from torch import Tensor, nn

from salt.models.transformer import DecoderLayer
from salt.stypes import Tensors
from salt.utils.tensor_utils import flatten_tensor_dict, masked_softmax


class Pooling(nn.Module):
    """Base class for pooling modules."""


class GlobalAttentionPooling(Pooling):
    """Global attention pooling over concatenated node embeddings.

    Uses a learned gating network to produce attention weights over the
    flattened inputs (concatenated along the sequence dimension), then
    computes the weighted sum. A padded token is appended to avoid ONNX
    issues when there are no tracks.

    Parameters
    ----------
    input_size : int
        Dimensionality of each node embedding feature vector.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = nn.Linear(input_size, 1)

    def forward(
        self,
        x: dict[str, Tensor] | dict,
        pad_mask: dict | None = None,
    ) -> Tensor:
        """Apply global attention pooling.

        Parameters
        ----------
        x : dict[str, Tensor] | dict
            Mapping from input stream name to tensor with shape ``[B, L_i, D]``.
            All non-``"objects"`` entries are concatenated along the sequence
            dimension to form a single ``[B, L, D]`` tensor.
        pad_mask : dict | None, optional
            Mapping from input stream name to boolean/byte mask of shape
            ``[B, L_i]``. Masks are concatenated along the sequence dimension
            and used to suppress padded positions. The default is ``None``.

        Returns
        -------
        Tensor
            Pooled tensor of shape ``[B, D]``.
        """
        x_flat = flatten_tensor_dict(x, exclude=["objects"])

        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1).unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x_flat), pad_mask, dim=1)
        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x_flat.shape[0], 1, x_flat.shape[2]), device=x_flat.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x_flat = torch.cat([x_flat, x_pad], dim=1)

        return (x_flat * weights).sum(dim=1)


class NodeQueryGAP(Pooling):
    """Global attention pooling over nodes and queries, then concatenation.

    Applies global attention pooling separately to (1) the concatenated node
    embeddings and (2) the object query embeddings, then concatenates both pooled
    vectors along the feature dimension.

    Parameters
    ----------
    input_size : int
        Dimensionality of each node/query embedding feature vector.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn_1 = nn.Linear(input_size, 1)
        self.gate_nn_2 = nn.Linear(input_size, 1)

    def forward(
        self,
        x: Tensors,
        pad_mask: dict | None = None,
    ) -> Tensor:
        """Apply global attention pooling to nodes and queries, then concatenate.

        Parameters
        ----------
        x : Tensors
            Mapping with at least:
            - non-``"objects"`` streams of shape ``[B, L_i, D]`` (nodes),
            - ``x["objects"]["embed"]`` of shape ``[B, M, D]`` (queries).
        pad_mask : dict | None, optional
            Mapping from stream name to padding mask ``[B, L_i]`` for node streams.
            The default is ``None``.

        Returns
        -------
        Tensor
            Concatenated pooled tensor of shape ``[B, 2D]`` consisting of
            ``[pooled_nodes, pooled_queries]``.
        """
        # Global Attention Pooling applied to both the decoder kv embeddings and
        # the encoder embeddings.
        x_nodes = flatten_tensor_dict(x, exclude=["objects"])

        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1).unsqueeze(-1)

        weights = masked_softmax(self.gate_nn_1(x_nodes), pad_mask, dim=1)

        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x_nodes.shape[0], 1, x_nodes.shape[2]), device=x_nodes.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x_nodes = torch.cat([x_nodes, x_pad], dim=1)
        pooled_nodes = (x_nodes * weights).sum(dim=1)

        # get pooled queries
        emb_queries = x["objects"]["embed"]
        query_pad = torch.zeros(
            (emb_queries.shape[0], 1, emb_queries.shape[2]),
            device=emb_queries.device,
        )
        padded_queries = torch.cat([emb_queries, query_pad], dim=1)
        weights = self.gate_nn_2(padded_queries).softmax(1)
        pooled_queries = (padded_queries * weights).sum(dim=1)

        # concatenate pooled nodes and queries
        return torch.cat([pooled_nodes, pooled_queries], dim=-1)


class ClassAttentionPooling(Pooling):
    """Class-attention pooling (ParT / DeParT / CaiT style).

    A single learned class token cross-attends to the encoded constituents
    through one or more :class:`~salt.models.transformer.DecoderLayer` blocks
    (cross-attention only, no self-attention, matching the CMS b-hive ParT
    class-attention head), and the resulting token is used as the pooled global
    representation. Following b-hive/CaiT, the class token is prepended to the
    keys/values of every block, so it is always a valid (non-padded) key and
    fully padded jets stay NaN-free (and ONNX-safe).

    Parameters
    ----------
    input_size : int
        Dimensionality of each constituent embedding (and of the class token).
    num_layers : int, optional
        Number of cross-attention blocks. The default is ``1``.
    ls_init : float | None, optional
        Initial LayerScale value for each block. The default is ``1e-3``.
    dense_kwargs : dict | None, optional
        Keyword args for the block's :class:`~salt.models.transformer.GLU` FFN.
    attn_kwargs : dict | None, optional
        Keyword args for the block's :class:`~salt.models.transformer.Attention`.
    norm : str, optional
        Normalization class name. The default is ``"LayerNorm"``.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int = 1,
        ls_init: float | None = 1e-3,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        norm: str = "LayerNorm",
    ):
        super().__init__()
        self.input_size = input_size
        # a single class token, initialised like the transformer register tokens
        self.class_token = nn.Parameter(torch.normal(torch.zeros((1, input_size)), std=1e-4))
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=input_size,
                ls_init=ls_init,
                dense_kwargs=dense_kwargs,
                attn_kwargs=attn_kwargs,
                norm=norm,
                self_attn=False,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: Tensors,
        pad_mask: dict | None = None,
    ) -> Tensor:
        """Apply class-attention pooling.

        Parameters
        ----------
        x : Tensors
            Mapping from input stream name to tensor with shape ``[B, L_i, D]``.
            All non-``"objects"`` entries are concatenated along the sequence
            dimension to form the ``[B, L, D]`` set of constituents.
        pad_mask : dict | None, optional
            Mapping from stream name to padding mask ``[B, L_i]`` (``True`` =
            padded). Masks are concatenated along the sequence dimension. The
            default is ``None``.

        Returns
        -------
        Tensor
            Pooled tensor of shape ``[B, D]``.
        """
        x_flat = flatten_tensor_dict(x, exclude=["objects"])

        mask = None
        if pad_mask is not None:
            mask = torch.cat(list(pad_mask.values()), dim=1)

        q = self.class_token.expand(x_flat.shape[0], -1, -1)
        for layer in self.layers:
            # prepend the class token to the keys/values (b-hive/CaiT class attention),
            # so it is always a valid key and fully padded jets stay NaN-free
            kv = torch.cat([q, x_flat], dim=1)
            kv_mask = None
            if mask is not None:
                valid = torch.zeros((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
                kv_mask = torch.cat([valid, mask], dim=1)
            q = layer(q, kv=kv, kv_mask=kv_mask)

        return q.squeeze(1)
