"""Read or patch Lingshu decoder activations without editing model source."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention


class ImagePriorAttention:
    """Capture image-query attention paid to the three prefix keys."""

    def __init__(self, image_positions: list[int], prior_positions=(0, 1, 2)):
        self.image_positions = image_positions
        self.prior_positions = list(prior_positions)
        self.blocks: dict[int, np.ndarray] = {}

    def observe(self, module, query, key, attention_mask, scaling):
        repeated_key = key.repeat_interleave(
            module.num_key_value_groups, dim=1
        )
        query_index = torch.as_tensor(
            self.image_positions, device=query.device
        )
        selected_query = query.index_select(-2, query_index).float()
        logits = selected_query @ repeated_key.float().transpose(-1, -2)
        logits *= scaling
        if attention_mask is not None:
            logits += attention_mask.index_select(
                -2, query_index
            )[..., : repeated_key.shape[-2]].float()
        else:
            keys = torch.arange(repeated_key.shape[-2], device=query.device)
            allowed = keys[None, :] <= query_index[:, None]
            logits.masked_fill_(~allowed[None, None], -torch.inf)
        probabilities = logits.softmax(dim=-1)[0].index_select(
            -1, torch.as_tensor(self.prior_positions, device=query.device)
        )
        self.blocks[module.layer_idx] = (
            probabilities.cpu().to(torch.float16).numpy()
        )

    @contextmanager
    def active(self):
        mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        original = mapping["sdpa"]

        def observed(module, query, key, value, attention_mask, **kwargs):
            if isinstance(module, Qwen2_5_VLAttention):
                self.observe(
                    module, query, key, attention_mask, kwargs["scaling"]
                )
            return original(
                module, query, key, value, attention_mask, **kwargs
            )

        mapping["sdpa"] = observed
        try:
            yield
        finally:
            mapping["sdpa"] = original

    def array(self) -> np.ndarray:
        layers = sorted(self.blocks)
        if layers != list(range(len(layers))):
            raise RuntimeError(f"Missing decoder layers: {layers}")
        return np.stack([self.blocks[layer] for layer in layers])


class HiddenStateCapture:
    """Capture selected positions from decoder-layer outputs."""

    def __init__(self, layers, regions: dict[str, list[int]]):
        self.layers = list(layers)
        self.regions = regions
        self.states: dict[int, dict[str, torch.Tensor]] = {}

    def hook(self, layer):
        def capture(_module, _inputs, output):
            hidden = output[0]
            self.states[layer] = {
                name: hidden[0, positions].detach().cpu().to(torch.bfloat16)
                for name, positions in self.regions.items()
            }
        return capture

    @contextmanager
    def active(self, decoder_layers):
        with ExitStack() as stack:
            for layer in self.layers:
                handle = decoder_layers[layer].register_forward_hook(
                    self.hook(layer)
                )
                stack.callback(handle.remove)
            yield
        if sorted(self.states) != sorted(self.layers):
            raise RuntimeError("Not all requested hidden states were captured")


class HiddenStatePatch:
    """Replace selected decoder-layer output positions during one forward."""

    def __init__(
        self,
        replacements: dict[int, tuple[list[int], torch.Tensor]],
    ):
        self.replacements = replacements

    def hook(self, layer):
        target_positions, replacement = self.replacements[layer]

        def patch(_module, _inputs, output):
            hidden = output[0].clone()
            source = replacement.to(hidden.device, dtype=hidden.dtype)
            if source.shape != hidden[0, target_positions].shape:
                raise RuntimeError(
                    f"Patch shape mismatch at layer {layer}: "
                    f"{source.shape} vs {hidden[0, target_positions].shape}"
                )
            hidden[0, target_positions] = source
            return (hidden, *output[1:])
        return patch

    @contextmanager
    def active(self, decoder_layers):
        with ExitStack() as stack:
            for layer in self.replacements:
                handle = decoder_layers[layer].register_forward_hook(
                    self.hook(layer)
                )
                stack.callback(handle.remove)
            yield
