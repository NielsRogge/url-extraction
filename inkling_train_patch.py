"""Memory-efficient training attention for transformers' Inkling 2 loop model (`inkling2_loop`).

transformers 5.24 implements Inkling attention eagerly only (per-token attention sinks and a
hidden-state-dependent relative position bias rule out SDPA/flash/flex). Eager attention materializes a dense
`[batch, heads, L, L]` relative position bias, an `[L, L]` distance matrix, causal masks, and fp32 scores, which
is ~100 GB per layer at L = 32k. This patch keeps the exact eager math but

- never builds the dense bias, distance matrix or masks (training only: no cache, batch size 1, no padding);
- computes attention in query chunks, each under activation checkpointing, building the bias for the chunk from
  the compact `[L, heads, d_rel]` relative states;
- restricts sliding-window layers (20 of 24, window 1024) to the keys inside their window, so they cost O(L * W).

Call `enable()` before running a forward pass on a model loaded for training. Inference with a cache, or with a
padded batch, falls through to the original implementation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_ENABLED = False
CHUNK_SIZE = 1024


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return states
    batch, heads, length, dim = states.shape
    return states[:, :, None].expand(batch, heads, groups, length, dim).reshape(batch, heads * groups, length, dim)


def _attention_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relative_states: torch.Tensor,
    proj: torch.Tensor,
    sinks: torch.Tensor,
    q_start: int,
    k_start: int,
    scaling: float,
    groups: int,
    rel_extent: int,
    window: int | None,
) -> torch.Tensor:
    """Eager Inkling attention for queries [q_start, q_start + c) against keys [k_start, k_start + kc)."""
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling  # [B, H, c, kc]
    q_positions = torch.arange(query.shape[2], device=query.device) + q_start
    k_positions = torch.arange(key.shape[2], device=query.device) + k_start
    distance = q_positions[:, None] - k_positions[None, :]  # [c, kc]
    # Relative position bias (as Inkling2LoopRelativePositionBias.forward, restricted to this chunk).
    rel_logits = (relative_states @ proj).transpose(1, 2)  # [B, H, c, R]
    gather_index = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    bias = rel_logits.gather(-1, gather_index)
    bias = bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)
    scores = scores + bias.to(scores.dtype)
    disallowed = distance < 0
    if window is not None:
        disallowed = disallowed | (distance >= window)
    scores = scores.masked_fill(disallowed, torch.finfo(scores.dtype).min)
    combined = torch.cat([scores, sinks], dim=-1).float()
    probabilities = F.softmax(combined, dim=-1)[..., :-1].to(query.dtype)
    return torch.matmul(probabilities, value)  # [B, H, c, D]


def chunked_attention(
    module: Any,
    hidden_states: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    bias_module = module.relative_position_bias
    relative_states = bias_module.linear(hidden_states).view(*hidden_states.shape[:-1], bias_module.num_heads, -1)
    relative_states = bias_module.norm(relative_states, position_ids)  # [B, L, H, d_rel]
    sinks = module.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)  # [B, H, L, 1]
    length = query.shape[2]
    window = module.sliding_window
    outputs = []
    for q_start in range(0, length, CHUNK_SIZE):
        q_end = min(length, q_start + CHUNK_SIZE)
        k_start = 0 if window is None else max(0, q_start - window + 1)
        args = (
            query[:, :, q_start:q_end],
            key[:, :, k_start:q_end],
            value[:, :, k_start:q_end],
            relative_states[:, q_start:q_end],
            bias_module.proj,
            sinks[:, :, q_start:q_end],
            q_start,
            k_start,
            module.scaling,
            module.num_key_value_groups,
            bias_module.rel_extent,
            window,
        )
        if torch.is_grad_enabled():
            outputs.append(checkpoint(_attention_chunk, *args, use_reentrant=False))
        else:
            outputs.append(_attention_chunk(*args))
    return torch.cat(outputs, dim=2)


def _usable(cache: Any, hidden_states: torch.Tensor, kwargs: dict[str, Any]) -> bool:
    return _ENABLED and cache is None and hidden_states.shape[0] == 1


def enable(chunk_size: int = CHUNK_SIZE) -> None:
    """Patch transformers' inkling2_loop modeling module in place (idempotent)."""
    global _ENABLED, CHUNK_SIZE
    from transformers.models.inkling2_loop import modeling_inkling2_loop as modeling

    CHUNK_SIZE = chunk_size
    if _ENABLED:
        return
    _ENABLED = True

    def attention_forward(self, hidden_states, key_hidden_states, value_hidden_states, conv_mask, cache_offset, kwargs):
        input_shape = hidden_states.shape[:-1]
        position_ids = kwargs["position_ids"]
        query = self.q_proj(hidden_states, position_ids)
        key = self.k_proj(key_hidden_states, cache=None, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
        value = self.v_proj(value_hidden_states, cache=None, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
        output = globals()["chunked_attention"](self, hidden_states, query, key, value, position_ids)
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output, hidden_states), None

    sliding_original = modeling.Inkling2LoopSlidingAttention.forward
    full_original = modeling.Inkling2LoopFullAttention.forward

    def sliding_forward(
        self, hidden_states, attention_mask, relative_distance, conv_mask=None, cache=None,
        key_hidden_states=None, value_hidden_states=None, cache_offset=0, **kwargs,
    ):
        if not _usable(cache, hidden_states, kwargs) or (relative_distance is not None and relative_distance.numel()):
            return sliding_original(
                self, hidden_states, attention_mask, relative_distance, conv_mask=conv_mask, cache=cache,
                key_hidden_states=key_hidden_states, value_hidden_states=value_hidden_states,
                cache_offset=cache_offset, **kwargs,
            )
        key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
        value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states
        return attention_forward(self, hidden_states, key_hidden_states, value_hidden_states, conv_mask, cache_offset, kwargs)

    def full_forward(
        self, hidden_states, key_hidden_states, value_hidden_states, attention_mask, relative_distance,
        conv_mask=None, cache=None, cache_offset=0, **kwargs,
    ):
        if not _usable(cache, hidden_states, kwargs) or (relative_distance is not None and relative_distance.numel()):
            return full_original(
                self, hidden_states, key_hidden_states, value_hidden_states, attention_mask, relative_distance,
                conv_mask=conv_mask, cache=cache, cache_offset=cache_offset, **kwargs,
            )
        return attention_forward(self, hidden_states, key_hidden_states, value_hidden_states, conv_mask, cache_offset, kwargs)

    modeling.Inkling2LoopSlidingAttention.forward = sliding_forward
    modeling.Inkling2LoopFullAttention.forward = full_forward

    # Skip the dense [L, L] distance matrices and 4D masks when training without a cache on one unpadded sequence.
    original_distance = modeling.get_relative_distance
    original_causal = modeling.create_causal_mask
    original_sliding = modeling.create_sliding_window_causal_mask

    def get_relative_distance(inputs_embeds, cache, is_sliding):
        if _ENABLED and cache is None and inputs_embeds.shape[0] == 1:
            return inputs_embeds.new_empty(0, dtype=torch.long)
        return original_distance(inputs_embeds, cache, is_sliding)

    def unpadded(attention_mask: torch.Tensor | None) -> bool:
        return attention_mask is None or bool(attention_mask.all())

    def create_causal_mask(**kwargs):
        if _ENABLED and kwargs.get("cache") is None and kwargs["inputs_embeds"].shape[0] == 1 and unpadded(kwargs.get("attention_mask")):
            return None
        return original_causal(**kwargs)

    def create_sliding_window_causal_mask(**kwargs):
        if _ENABLED and kwargs.get("cache") is None and kwargs["inputs_embeds"].shape[0] == 1 and unpadded(kwargs.get("attention_mask")):
            return None
        return original_sliding(**kwargs)

    modeling.get_relative_distance = get_relative_distance
    modeling.create_causal_mask = create_causal_mask
    modeling.create_sliding_window_causal_mask = create_sliding_window_causal_mask


def disable() -> None:
    """Route subsequent calls back to the original implementation (patched functions stay installed)."""
    global _ENABLED
    _ENABLED = False
