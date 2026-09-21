"""Representation-shift and statistical utilities for mechanism diagnosis."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def shift_metrics(base: torch.Tensor, other: torch.Tensor):
    base = base.float()
    other = other.float()
    if base.shape != other.shape:
        raise ValueError(f"Hidden-state shape mismatch: {base.shape}/{other.shape}")
    eps = 1e-8

    def vector_metrics(a, b):
        cosine = 1 - torch.nn.functional.cosine_similarity(
            a, b, dim=-1, eps=eps
        )
        scale = (a.norm(dim=-1) + b.norm(dim=-1)) / 2
        normalized_l2 = (a - b).norm(dim=-1) / scale.clamp_min(eps)
        return float(cosine.mean()), float(normalized_l2.mean())

    mean_cos, mean_l2 = vector_metrics(
        base.mean(dim=0, keepdim=True),
        other.mean(dim=0, keepdim=True),
    )
    max_cos, max_l2 = vector_metrics(
        base.max(dim=0, keepdim=True).values,
        other.max(dim=0, keepdim=True).values,
    )
    token_cos, token_l2 = vector_metrics(base, other)
    return np.asarray(
        [mean_cos, mean_l2, max_cos, max_l2, token_cos, token_l2],
        dtype=np.float32,
    )


SHIFT_NAMES = (
    "mean_cosine", "mean_normalized_l2",
    "max_cosine", "max_normalized_l2",
    "token_cosine", "token_normalized_l2",
)


def correlation(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0
    result = spearmanr(x, y)
    rho = float(result.statistic)
    p = float(result.pvalue)
    return (
        rho if np.isfinite(rho) else 0.0,
        p if np.isfinite(p) else 1.0,
    )


def fdr_bh(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    flat = values.reshape(-1)
    order = np.argsort(flat)
    adjusted = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    output = np.empty_like(flat)
    output[order] = adjusted
    return output.reshape(values.shape)


def bootstrap_mean_ci(values, rng, repeats=2000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
