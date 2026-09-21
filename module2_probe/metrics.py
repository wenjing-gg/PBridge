"""Paired spatial-attention statistics; all distributions are conditional on vision."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def normalize(block: np.ndarray) -> np.ndarray:
    values = block.astype(np.float32)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)


def spatial_metrics(base: np.ndarray, prior: np.ndarray):
    # [layers, heads, question, image] -> [layers, heads, image]
    if base.shape != prior.shape:
        raise ValueError(f"Unmatched spatial attention shapes: {base.shape} vs {prior.shape}")
    a = normalize(base.mean(axis=-2))
    b = normalize(prior.mean(axis=-2))
    mid = (a + b) / 2
    js = (entropy(mid) - (entropy(a) + entropy(b)) / 2).clip(0)
    l1 = abs(a - b).sum(axis=-1)
    delta_entropy = entropy(b) - entropy(a)
    overlap = {}
    for k in (5, 10, 20):
        top_a = np.argsort(a, axis=-1)[..., -k:]
        top_b = np.argsort(b, axis=-1)[..., -k:]
        overlap[k] = (top_a[..., :, None] == top_b[..., None, :]).any(
            axis=-1
        ).sum(axis=-1) / k
    return {
        "js": js, "l1": l1, "delta_entropy": delta_entropy,
        "entropy_base": entropy(a), "entropy_prior": entropy(b),
        **{f"top{k}_overlap": value for k, value in overlap.items()},
    }


def correlation(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0
    result = spearmanr(x, y)
    rho = float(result.statistic)
    p = float(result.pvalue)
    return rho if np.isfinite(rho) else 0.0, p if np.isfinite(p) else 1.0
