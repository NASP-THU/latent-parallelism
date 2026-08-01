# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Tensor Parallel for WanModel inference.
#
# Shards each WanAttentionBlock across TP ranks:
#   - Self-attention q/k/v: column-parallel (split heads)
#   - Self-attention o: row-parallel (all-reduce after)
#   - Cross-attention q/k/v/o: same pattern
#   - FFN first linear: column-parallel
#   - FFN second linear: row-parallel (all-reduce after)
#
# Everything else (embeddings, head, norms, modulation) is replicated.

import logging
import types

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Weight slicing helpers
# ═══════════════════════════════════════════════════════════════════════

def _shard_linear_col(linear: nn.Linear, tp_rank: int, tp_size: int):
    """Column-parallel: slice output dimension of a Linear layer in-place."""
    assert linear.out_features % tp_size == 0, (
        f"out_features ({linear.out_features}) not divisible by tp_size ({tp_size})")
    shard = linear.out_features // tp_size
    s, e = tp_rank * shard, (tp_rank + 1) * shard
    linear.weight = nn.Parameter(linear.weight.data[s:e].contiguous())
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.data[s:e].contiguous())
    linear.out_features = shard


def _shard_linear_row(linear: nn.Linear, tp_rank: int, tp_size: int):
    """Row-parallel: slice input dimension of a Linear layer in-place.
    Bias is divided by tp_size so all-reduce produces correct result."""
    assert linear.in_features % tp_size == 0, (
        f"in_features ({linear.in_features}) not divisible by tp_size ({tp_size})")
    shard = linear.in_features // tp_size
    s, e = tp_rank * shard, (tp_rank + 1) * shard
    linear.weight = nn.Parameter(linear.weight.data[:, s:e].contiguous())
    if linear.bias is not None:
        linear.bias = nn.Parameter((linear.bias.data / tp_size).contiguous())
    linear.in_features = shard


def _shard_rmsnorm(norm: nn.Module, tp_rank: int, tp_size: int):
    """Slice RMSNorm weight to match column-parallel output dimension."""
    if isinstance(norm, nn.Identity) or not hasattr(norm, 'weight'):
        return
    dim = norm.weight.shape[0]
    assert dim % tp_size == 0, (
        f"norm dim ({dim}) not divisible by tp_size ({tp_size})")
    shard = dim // tp_size
    s, e = tp_rank * shard, (tp_rank + 1) * shard
    norm.weight = nn.Parameter(norm.weight.data[s:e].contiguous())
    if hasattr(norm, 'dim'):
        norm.dim = shard


def _shard_attention(attn: nn.Module, tp_rank: int, tp_size: int):
    """Shard attention: q/k/v column-parallel, o row-parallel."""
    _shard_linear_col(attn.q, tp_rank, tp_size)
    _shard_linear_col(attn.k, tp_rank, tp_size)
    _shard_linear_col(attn.v, tp_rank, tp_size)
    _shard_linear_row(attn.o, tp_rank, tp_size)
    _shard_rmsnorm(attn.norm_q, tp_rank, tp_size)
    _shard_rmsnorm(attn.norm_k, tp_rank, tp_size)
    # I2V extra projections (if present)
    if hasattr(attn, 'k_img'):
        _shard_linear_col(attn.k_img, tp_rank, tp_size)
    if hasattr(attn, 'v_img'):
        _shard_linear_col(attn.v_img, tp_rank, tp_size)
    if hasattr(attn, 'norm_k_img'):
        _shard_rmsnorm(attn.norm_k_img, tp_rank, tp_size)
    attn.num_heads = attn.num_heads // tp_size
    attn.dim = attn.dim // tp_size


def _shard_ffn(ffn: nn.Sequential, tp_rank: int, tp_size: int):
    """Shard FFN: first Linear column-parallel, second Linear row-parallel."""
    _shard_linear_col(ffn[0], tp_rank, tp_size)   # Linear(dim, ffn_dim)
    _shard_linear_row(ffn[2], tp_rank, tp_size)   # Linear(ffn_dim, dim)


# ═══════════════════════════════════════════════════════════════════════
#  Monkey-patched block forward with all-reduce
# ═══════════════════════════════════════════════════════════════════════

def _make_tp_block_forward(tp_group):
    """Return a patched WanAttentionBlock.forward that inserts all-reduce."""

    def tp_block_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention + all-reduce
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens, grid_sizes, freqs)
        dist.all_reduce(y, group=tp_group)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention + all-reduce
        cross_out = self.cross_attn(self.norm3(x), context, context_lens)
        dist.all_reduce(cross_out, group=tp_group)
        x = x + cross_out

        # ffn + all-reduce
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        dist.all_reduce(y, group=tp_group)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

        return x

    return tp_block_forward


# ═══════════════════════════════════════════════════════════════════════
#  Setup function
# ═══════════════════════════════════════════════════════════════════════

def setup_tensor_parallel(model, tp_rank, tp_size, device):
    """
    Apply Tensor Parallelism to a WanModel instance.

    1. Shards attention (q/k/v col-parallel, o row-parallel) and FFN
       (first linear col-parallel, second linear row-parallel) in every block.
    2. Monkey-patches each block's forward to insert all-reduce after
       self-attention, cross-attention, and FFN.
    3. Moves the sharded model to device.

    Args:
        model:   WanModel instance (weights on CPU).
        tp_rank: This rank's TP index (0 .. tp_size-1).
        tp_size: Total number of TP ranks.
        device:  torch.device to place the sharded model on.

    Returns:
        The modified model (in-place).
    """
    assert tp_size > 1
    assert model.num_heads % tp_size == 0, (
        f"num_heads ({model.num_heads}) must be divisible by "
        f"tp_size ({tp_size})")

    logger.info(
        f"TP rank {tp_rank}/{tp_size}: sharding {len(model.blocks)} blocks "
        f"({model.num_heads} heads -> {model.num_heads // tp_size} per rank)")

    tp_group = dist.group.WORLD

    for block in model.blocks:
        _shard_attention(block.self_attn, tp_rank, tp_size)
        _shard_attention(block.cross_attn, tp_rank, tp_size)
        _shard_ffn(block.ffn, tp_rank, tp_size)
        block.forward = types.MethodType(
            _make_tp_block_forward(tp_group), block)

    model.to(device)
    return model
