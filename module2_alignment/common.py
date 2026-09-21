"""Frozen Module 1 setup and deterministic Module 2 data split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from module2_probe.probe_lingshu import (
    FrozenReader,
    answer_indices,
    load_ppr,
    load_rows,
)
from module2_probe.fixed_module1 import sha256, validate_fixed_module1


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "module2_alignment" / "outputs"
CHECKPOINT = ROOT / "module1_ppr" / "artifacts" / "lingshu_7b" / "ppr_moe.pt"
SEED = 20260920


def split_ids(rows):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    dev = np.sort(order[: round(len(rows) * 0.2)])
    train = np.sort(order[round(len(rows) * 0.2) :])
    return {"train": train, "dev": dev}


def margin(logits, gold):
    logits = np.asarray(logits, dtype=np.float64)
    other = logits.copy()
    other[np.arange(len(gold)), gold] = -np.inf
    return logits[np.arange(len(gold)), gold] - other.max(axis=1)


def frozen_ppr(device):
    reader = FrozenReader(device)
    data, scores, ranks, prefixes = load_ppr(reader, CHECKPOINT, "train")
    rows = load_rows("train")
    assert len(rows) == len(data["query_embeddings"])
    return reader, data, scores, ranks, prefixes, rows, answer_indices(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checkpoint_hash():
    validate_fixed_module1(CHECKPOINT)
    return sha256(CHECKPOINT)
