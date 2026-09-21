"""Post-hoc contiguous-layer patching after the prespecified single-layer test."""

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


GROUPS = {
    "early_0_2": [0, 1, 2],
    "late_25_26": [25, 26],
    "late_25_27": [25, 26, 27],
}


def capture(reader, row, prefix, layers, regions):
    probe = HiddenStateCapture(layers, regions)
    decoder = reader.model.model.language_model.layers
    with torch.no_grad(), probe.active(decoder):
        logits = reader.logits(
            [row], [(0, [])], prefixes=prefix
        )[0].float().cpu().numpy()
    return logits, probe.states


def patched(reader, row, prefix, replacements):
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("semantic", "prior"),
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("module2_probe/outputs/mechanism/patching"),
    )
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    before = module1_hashes()
    reader = FrozenReader(args.device)
    data, _, ranks, prefixes = load_ppr(
        reader, args.checkpoint, "test"
    )
    rows = load_rows("test")
    gold = answer_indices(rows)
    controls_for = control_prefixes(
        reader, args.checkpoint, data, ranks, prefixes, args.seed
    )
    all_layers = sorted({layer for group in GROUPS.values() for layer in group})
    semantic_kinds = ("image", "question", "options")
    prior_kinds = ("prior_random", "prior_shuffled")
    kinds = semantic_kinds if args.mode == "semantic" else prior_kinds
    output_path = args.save_dir / f"intervals_{args.mode}.jsonl"
    started = time.monotonic()
    count = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for index in range(300, len(rows)):
            row = rows[index]
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
            prefixes_by_condition, _ = controls_for(index)
            base_states = {}
            control_states = {}
            if args.mode == "semantic":
                _, base_states = capture(
                    reader,
                    row,
                    None,
                    all_layers,
                    {
                        region: ordinary[region]
                        for region in semantic_kinds
                    },
                )
            else:
                for condition in ("random", "shuffled"):
                    _, control_states[condition] = capture(
                        reader,
                        row,
                        prefixes_by_condition[condition],
                        all_layers,
                        {"prior": [0, 1, 2]},
                    )
            correct_prefix = prefixes_by_condition["correct"]
            correct_logits = reader.logits(
                [row], [(0, [])], prefixes=correct_prefix
            )[0].float().cpu().numpy()
            correct_margin = margin(correct_logits, int(gold[index]))
            results = {}
            for kind in kinds:
                results[kind] = {}
                for group_name, group_layers in GROUPS.items():
                    replacements = {}
                    for layer in group_layers:
                        if args.mode == "semantic":
                            positions = shifted[kind]
                            replacement = base_states[layer][kind]
                        else:
                            condition = kind.removeprefix("prior_")
                            positions = shifted["prior"]
                            replacement = control_states[condition][layer][
                                "prior"
                            ]
                        replacements[layer] = (
                            positions,
                            replacement,
                        )
                    logits = patched(
                        reader, row, correct_prefix, replacements
                    )
                    patched_margin = margin(
                        logits, int(gold[index])
                    )
                    results[kind][group_name] = {
                        "prediction": "ABCD"[int(logits.argmax())],
                        "margin": patched_margin,
                        "recovery_loss": (
                            correct_margin - patched_margin
                        ),
                    }
            record = {
                "index": index,
                "id": str(row["id"]),
                "gold": str(row["answer"]),
                "no_prior_prediction": "ABCD"[
                    int(
                        reader.logits(
                            [row], [(0, [])]
                        )[0].float().cpu().numpy().argmax()
                    )
                ],
                "correct_prediction": "ABCD"[
                    int(correct_logits.argmax())
                ],
                "correct_margin": correct_margin,
                "patches": results,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            count += 1
            if count == 1 or count % 10 == 0 or count == 113:
                print(
                    f"interval-{args.mode} {count}/113 "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if count != 113:
        raise RuntimeError(f"Expected 113 cases, got {count}")
    if module1_hashes() != before:
        raise RuntimeError("Module 1 source files changed")
    reader.close()


if __name__ == "__main__":
    main()
