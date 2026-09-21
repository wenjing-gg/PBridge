"""Reuse the frozen Module 1 reader and discover precise multimodal token spans."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module1_ppr"))

from module1_ppr.reader import FrozenReader, load_image, option_values
from module1_ppr.run import answer_indices, load_assets, load_rows
from module1_ppr.model import PPRMoE, TOP_CONTEXT

from module2_probe.attention_hooks import AttentionProbe, Intervention
from module2_probe.fixed_module1 import validate_fixed_module1


_offset_tokenizer = None


def spans(reader: FrozenReader, row: dict, prefix_length: int = 0):
    global _offset_tokenizer
    if _offset_tokenizer is None:
        _offset_tokenizer = AutoTokenizer.from_pretrained(
            str(reader.model_dir), local_files_only=True, use_fast=True
        )
    inputs, images = reader._prepare([row], [(0, [])])
    for image in images:
        image.close()
    image_id = int(reader.processor.image_token_id)
    ids = inputs["input_ids"][0].tolist()
    image_positions = [index for index, value in enumerate(ids) if value == image_id]
    grid = inputs["image_grid_thw"][0].tolist()
    expected = int(np.prod(grid)) // int(reader.processor.image_processor.merge_size) ** 2
    if len(image_positions) != expected or image_positions != list(
        range(image_positions[0], image_positions[0] + expected)
    ):
        raise RuntimeError("Noncontiguous image tokens or processor grid mismatch")

    image = load_image(row["image"])
    try:
        text = reader.processor.apply_chat_template(
            [[{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": reader.prompt(
                    str(row["question"]), option_values(row), []
                )},
            ]}]],
            tokenize=False, add_generation_prompt=True,
        )[0]
    finally:
        image.close()
    encoded = _offset_tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = []
    expanded = []
    for token, offset in zip(encoded["input_ids"], encoded["offset_mapping"]):
        repeat = len(image_positions) if token == image_id else 1
        expanded.extend([token] * repeat)
        offsets.extend([offset] * repeat)
    if expanded != ids:
        raise RuntimeError("Chat template/token offsets do not match processor input_ids")
    question_start = text.index("Question: ") + len("Question: ")
    question_end = question_start + len(str(row["question"]))
    options_start = text.index("Options:\n", question_end) + len("Options:\n")
    options_end = text.index("\n\nRespond with", options_start)
    token_range = lambda start, end: [
        index + prefix_length
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    question = token_range(question_start, question_end)
    options = token_range(options_start, options_end)
    if not question or not options or max(question) >= min(options):
        raise RuntimeError("Invalid question/option token offsets")
    return {
        "image": [position + prefix_length for position in image_positions],
        "question": question,
        "options": options,
        "last": len(ids) - 1 + prefix_length,
        "grid": grid,
        "input_length": len(ids) + prefix_length,
    }


def load_ppr(reader: FrozenReader, checkpoint_path: Path, split: str):
    baseline = validate_fixed_module1(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["reader_dim"] != reader.hidden_size or checkpoint["reader"] != reader.reader_name:
        raise RuntimeError("Checkpoint does not match frozen reader")
    if not checkpoint["stage1"]["converged"] or not checkpoint["stage2"]["converged"]:
        raise RuntimeError("Module 1 checkpoint is not converged")
    if checkpoint.get("prefix_weighting") != baseline["prefix_weighting"]:
        raise RuntimeError("Checkpoint is not the fixed PBridge-B baseline")
    ppr = PPRMoE(reader.hidden_size).to(reader.model_device)
    ppr.load_state_dict(checkpoint["state_dict"])
    ppr.eval()
    data = load_assets(split)
    with torch.no_grad():
        query = torch.from_numpy(data["query_embeddings"]).to(reader.model_device)
        knowledge = torch.from_numpy(data["candidate_embeddings"]).to(reader.model_device)
        utility = torch.from_numpy(data["utility_proxy"]).to(reader.model_device)
        scores, _, _ = ppr(query, knowledge, utility)
        ranks = scores.topk(TOP_CONTEXT, dim=-1).indices
        prefix = ppr.prefix_embeddings(knowledge, scores, ranks)
    return data, scores.cpu().numpy(), ranks.cpu().numpy(), prefix


def run(reader, row, prefix, positions, intervention: Intervention | None = None, capture=True):
    probe = AttentionProbe(
        positions["question"], positions["image"], positions["last"], intervention
    )
    with torch.no_grad(), probe.active(capture=capture):
        logits = reader.logits([row], [(0, [])], prefixes=prefix)
    return logits[0].float().cpu().numpy(), probe.arrays() if capture else None


def margin(logits: np.ndarray, gold: int) -> float:
    alternatives = np.delete(logits, gold)
    return float(logits[gold] - alternatives.max())
