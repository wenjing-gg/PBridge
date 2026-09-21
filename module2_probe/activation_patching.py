"""Patch semantic regions in Correct-Prior runs on the held-out 113 cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from module2_probe.mechanism_common import control_prefixes, module1_hashes
from module2_probe.mechanism_hooks import HiddenStateCapture, HiddenStatePatch
from module2_probe.probe_lingshu import (
    FrozenReader,
    answer_indices,
    load_ppr,
    load_rows,
    margin,
    spans,
)


def capture(reader, row, prefix, layers, regions):
    probe = HiddenStateCapture(layers, regions)
    decoder = reader.model.model.language_model.layers
    with torch.no_grad(), probe.active(decoder):
        logits = reader.logits(
            [row], [(0, [])], prefixes=prefix
        )[0].float().cpu().numpy()
    return logits, probe.states


def patched_logits(reader, row, prefix, replacements):
    patch = HiddenStatePatch(replacements)
    decoder = reader.model.model.language_model.layers
    with torch.no_grad(), patch.active(decoder):
        return reader.logits(
            [row], [(0, [])], prefixes=prefix
        )[0].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("module1_ppr/artifacts/lingshu_7b/ppr_moe.pt"),
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=Path(
            "module2_probe/outputs/mechanism/analysis/"
            "candidate_layers.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--patch-kinds",
        default="image,question,options",
        help="image,question,options,prior_random,prior_shuffled",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("module2_probe/outputs/mechanism/patching"),
    )
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    kinds = tuple(value.strip() for value in args.patch_kinds.split(","))
    valid = {
        "image", "question", "options",
        "prior_random", "prior_shuffled",
    }
    if not set(kinds) <= valid:
        raise ValueError(f"Unknown patch kinds: {set(kinds) - valid}")
    layers = json.loads(
        args.candidate_file.read_text(encoding="utf-8")
    )["layers"]
    before = module1_hashes()
    reader = FrozenReader(args.device)
    data, scores, ranks, prefixes = load_ppr(
        reader, args.checkpoint, "test"
    )
    rows = load_rows("test")
    gold = answer_indices(rows)
    controls_for = control_prefixes(
        reader, args.checkpoint, data, ranks, prefixes, args.seed
    )
    output_name = "_".join(kinds) + ".jsonl"
    started = time.monotonic()
    processed = 0
    with (args.save_dir / output_name).open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            prefixes_by_condition, _ = controls_for(index)
            if index < 300:
                continue
            ordinary = spans(reader, row)
            shifted = {
                **ordinary,
                "image": [value + 3 for value in ordinary["image"]],
                "question": [
                    value + 3 for value in ordinary["question"]
                ],
                "options": [value + 3 for value in ordinary["options"]],
                "prior": [0, 1, 2],
            }
            base_regions = {
                region: ordinary[region]
                for region in ("image", "question", "options")
            }
            base_logits, base_states = capture(
                reader, row, None, layers, base_regions
            )
            correct_prefix = prefixes_by_condition["correct"]
            correct_logits = reader.logits(
                [row], [(0, [])], prefixes=correct_prefix
            )[0].float().cpu().numpy()
            control_states = {}
            for condition in ("random", "shuffled"):
                required = f"prior_{condition}" in kinds
                if required:
                    _, control_states[condition] = capture(
                        reader,
                        row,
                        prefixes_by_condition[condition],
                        layers,
                        {"prior": [0, 1, 2]},
                    )
            correct_margin = margin(
                correct_logits, int(gold[index])
            )
            results = {}
            for kind in kinds:
                results[kind] = {}
                for layer in layers:
                    if kind in ("image", "question", "options"):
                        positions = shifted[kind]
                        replacement = base_states[layer][kind]
                    else:
                        control = kind.removeprefix("prior_")
                        positions = shifted["prior"]
                        replacement = control_states[control][layer][
                            "prior"
                        ]
                    logits = patched_logits(
                        reader,
                        row,
                        correct_prefix,
                        {layer: (positions, replacement)},
                    )
                    patched_margin = margin(
                        logits, int(gold[index])
                    )
                    results[kind][str(layer)] = {
                        "logits": logits.tolist(),
                        "prediction": "ABCD"[int(logits.argmax())],
                        "margin": patched_margin,
                        "recovery_loss": correct_margin - patched_margin,
                    }
            record = {
                "index": index,
                "id": str(row["id"]),
                "gold": str(row["answer"]),
                "layers": layers,
                "no_prior_prediction": "ABCD"[
                    int(base_logits.argmax())
                ],
                "correct_prediction": "ABCD"[
                    int(correct_logits.argmax())
                ],
                "no_prior_margin": margin(
                    base_logits, int(gold[index])
                ),
                "correct_margin": correct_margin,
                "delta_margin": correct_margin
                - margin(base_logits, int(gold[index])),
                "patches": results,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            processed += 1
            if processed == 1 or processed % 10 == 0 or processed == 113:
                print(
                    f"patch {output_name}: {processed}/113 "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if processed != 113:
        raise RuntimeError(f"Expected 113 validation cases, got {processed}")
    if module1_hashes() != before:
        raise RuntimeError("Module 1 source files changed during patching")
    reader.close()


if __name__ == "__main__":
    main()
