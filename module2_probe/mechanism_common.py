"""Shared frozen PBridge setup for Module 2 mechanism experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from module1_ppr.gme_assets import KNOWLEDGE_EMBEDDINGS
from module1_ppr.model import PPRMoE
from module2_probe.fixed_module1 import module1_hashes


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def control_prefixes(
    reader,
    checkpoint_path,
    data,
    ranks,
    prefixes,
    seed,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = PPRMoE(reader.hidden_size).to(reader.model_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    bank = np.load(KNOWLEDGE_EMBEDDINGS, mmap_mode="r")
    rng = np.random.default_rng(seed)

    def controls(index):
        selected_utility = data["utility_proxy"][index, ranks[index]]
        donor = int(
            (index + 1 + rng.integers(len(prefixes) - 1)) % len(prefixes)
        )
        random_ids = rng.choice(len(bank), size=3, replace=False)
        random_knowledge = torch.as_tensor(
            np.asarray(bank[random_ids], dtype=np.float32),
            device=reader.model_device,
        )[None]
        query = torch.as_tensor(
            data["query_embeddings"][index],
            device=reader.model_device,
        )[None]
        proxy = torch.as_tensor(
            selected_utility,
            device=reader.model_device,
        )[None]
        with torch.no_grad():
            random_scores, _, _ = model(query, random_knowledge, proxy)
            random_ranks = random_scores.topk(3, dim=-1).indices
            random_prefix = model.prefix_embeddings(
                random_knowledge,
                random_scores,
                random_ranks,
            )
        return {
            "correct": prefixes[index : index + 1],
            "random": random_prefix,
            "shuffled": prefixes[donor : donor + 1],
        }, {
            "random_knowledge_ids": random_ids.tolist(),
            "shuffled_donor_index": donor,
        }

    return controls
