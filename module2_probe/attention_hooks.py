"""Observe Lingshu SDPA without changing its output; optionally intervene on one head."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention


@dataclass(frozen=True)
class Intervention:
    layer: int
    head: int
    image_positions: tuple[int, ...]


class AttentionProbe:
    def __init__(
        self,
        question_positions: list[int],
        image_positions: list[int],
        last_position: int,
        intervention: Intervention | None = None,
    ):
        self.question_positions = question_positions
        self.image_positions = image_positions
        self.last_position = last_position
        self.intervention = intervention
        self.blocks: dict[int, np.ndarray] = {}
        self.last_blocks: dict[int, np.ndarray] = {}
        self.image_mass: dict[int, np.ndarray] = {}
        self.layers_seen: set[int] = set()

    def observe(self, module, query, key, attention_mask, scaling):
        layer = module.layer_idx
        key = key.repeat_interleave(module.num_key_value_groups, dim=1)
        indices = self.question_positions + [self.last_position]
        selected = query[:, :, indices, :].float()
        logits = selected @ key.float().transpose(-1, -2)
        logits *= scaling
        if attention_mask is not None:
            logits += attention_mask[:, :, indices, : key.shape[-2]].float()
        else:
            positions = torch.arange(key.shape[-2], device=query.device)
            allowed = positions[None, :] <= torch.as_tensor(
                indices, device=query.device
            )[:, None]
            logits.masked_fill_(~allowed[None, None], -torch.inf)
        probabilities = logits.softmax(dim=-1)[0].index_select(
            -1, torch.as_tensor(self.image_positions, device=query.device)
        )
        question = probabilities[:, : len(self.question_positions)]
        self.blocks[layer] = question.cpu().to(torch.float16).numpy()
        self.last_blocks[layer] = probabilities[:, -1].cpu().to(torch.float16).numpy()
        self.image_mass[layer] = question.float().sum(dim=-1).mean(dim=-1).cpu().numpy()
        self.layers_seen.add(layer)

    def intervene(self, module, query, key, value, attention_mask, original, kwargs):
        change = self.intervention
        if change is None or module.layer_idx != change.layer:
            return original(module, query, key, value, attention_mask, **kwargs)
        output, weights = original(module, query, key, value, attention_mask, **kwargs)
        group = change.head // module.num_key_value_groups
        sequence = query.shape[-2]
        keys = key.shape[-2]
        if attention_mask is None:
            allowed = torch.arange(keys, device=query.device)[None, :] <= (
                torch.arange(sequence, device=query.device)[:, None]
            )
            allowed[:, list(change.image_positions)] = False
            local_mask = allowed[None, None]
        else:
            local_mask = attention_mask[:, :, :, :keys].clone()
            local_mask[..., list(change.image_positions)] = torch.finfo(
                local_mask.dtype
            ).min
        replaced = F.scaled_dot_product_attention(
            query[:, change.head : change.head + 1],
            key[:, group : group + 1],
            value[:, group : group + 1],
            attn_mask=local_mask,
            dropout_p=0.0,
            scale=kwargs["scaling"],
            is_causal=False,
        )
        output = output.clone()
        output[:, :, change.head, :] = replaced[:, 0]
        return output, weights

    def arrays(self):
        layers = sorted(self.layers_seen)
        if layers != list(range(len(layers))):
            raise RuntimeError(f"Nonconsecutive attention layers: {layers}")
        return {
            "q_image": np.stack([self.blocks[layer] for layer in layers]),
            "last_image": np.stack([self.last_blocks[layer] for layer in layers]),
            "image_mass": np.stack([self.image_mass[layer] for layer in layers]),
        }

    @contextmanager
    def active(self, capture: bool = True):
        mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        original = mapping["sdpa"]

        def observed(module, query, key, value, attention_mask, **kwargs):
            if isinstance(module, Qwen2_5_VLAttention):
                if capture:
                    self.observe(
                        module, query, key, attention_mask, kwargs["scaling"]
                    )
                return self.intervene(
                    module, query, key, value, attention_mask, original, kwargs
                )
            return original(module, query, key, value, attention_mask, **kwargs)

        mapping["sdpa"] = observed
        try:
            yield
        finally:
            mapping["sdpa"] = original
