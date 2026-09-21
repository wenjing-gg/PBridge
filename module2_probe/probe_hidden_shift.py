"""Measure layer-wise image/question/option representation shifts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from module2_probe.mechanism_common import (
    control_prefixes,
    module1_hashes,
    write_json,
)
from module2_probe.mechanism_hooks import HiddenStateCapture
from module2_probe.mechanism_metrics import SHIFT_NAMES, shift_metrics
from module2_probe.probe_lingshu import (
    FrozenReader,
    answer_indices,
    load_ppr,
    load_rows,
    margin,
    spans,
)


REGIONS = ("image", "question", "options")
CONDITIONS = ("correct", "random", "shuffled")


def capture(reader, row, prefix, positions):
    probe = HiddenStateCapture(
        range(28), {region: positions[region] for region in REGIONS}
    )
    layers = reader.model.model.language_model.layers
    with torch.no_grad(), probe.active(layers):
        logits = reader.logits(
            [row], [(0, [])], prefixes=prefix
        )[0].float().cpu().numpy()
    return logits, probe.states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("module1_ppr/artifacts/lingshu_7b/ppr_moe.pt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("module2_probe/outputs/mechanism/hidden_shift"),
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "cases").mkdir(exist_ok=True)
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
    parity = []
    started = time.monotonic()
    count = len(rows) if args.max_cases <= 0 else min(
        len(rows), args.max_cases
    )
    with (args.save_dir / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows[:count]):
            ordinary = spans(reader, row)
            shifted = {
                **ordinary,
                "image": [value + 3 for value in ordinary["image"]],
                "question": [value + 3 for value in ordinary["question"]],
                "options": [value + 3 for value in ordinary["options"]],
                "last": ordinary["last"] + 3,
            }
            prefixes_by_condition, control_metadata = controls_for(index)
            base_logits, base_states = capture(
                reader, row, None, ordinary
            )
            all_metrics = np.empty(
                (
                    len(CONDITIONS),
                    len(REGIONS),
                    28,
                    len(SHIFT_NAMES),
                ),
                dtype=np.float32,
            )
            logits_by_condition = {}
            for condition_index, condition in enumerate(CONDITIONS):
                logits, states = capture(
                    reader,
                    row,
                    prefixes_by_condition[condition],
                    shifted,
                )
                logits_by_condition[condition] = logits
                for region_index, region in enumerate(REGIONS):
                    for layer in range(28):
                        all_metrics[
                            condition_index, region_index, layer
                        ] = shift_metrics(
                            base_states[layer][region],
                            states[layer][region],
                        )
                del states
            if index < 5:
                native_base = reader.logits(
                    [row], [(0, [])]
                )[0].float().cpu().numpy()
                error = float(np.max(abs(native_base - base_logits)))
                parity.append(
                    {
                        "index": index,
                        "condition": "no_prior",
                        "max_abs_logit_error": error,
                        "same_prediction": bool(
                            native_base.argmax() == base_logits.argmax()
                        ),
                    }
                )
                for condition in CONDITIONS:
                    native = reader.logits(
                        [row],
                        [(0, [])],
                        prefixes=prefixes_by_condition[condition],
                    )[0].float().cpu().numpy()
                    error = float(
                        np.max(abs(native - logits_by_condition[condition]))
                    )
                    parity.append(
                        {
                            "index": index,
                            "condition": condition,
                            "max_abs_logit_error": error,
                            "same_prediction": bool(
                                native.argmax()
                                == logits_by_condition[condition].argmax()
                            ),
                        }
                    )
                if any(
                    record["max_abs_logit_error"] > 1e-3
                    or not record["same_prediction"]
                    for record in parity
                ):
                    raise RuntimeError("Hidden capture changed model output")
            np.savez_compressed(
                args.save_dir / "cases" / f"{index:04d}.npz",
                shift=all_metrics,
            )
            record = {
                "index": index,
                "id": str(row["id"]),
                "gold": str(row["answer"]),
                "no_prior_logits": base_logits.tolist(),
                "no_prior_prediction": "ABCD"[int(base_logits.argmax())],
                "no_prior_margin": margin(base_logits, int(gold[index])),
                "prior_ids": data["candidate_indices"][
                    index, ranks[index]
                ].tolist(),
                "prior_utility": data["utility_proxy"][
                    index, ranks[index]
                ].tolist(),
                "prior_score": scores[index, ranks[index]].tolist(),
                "image_positions_base": [
                    ordinary["image"][0],
                    ordinary["image"][-1],
                ],
                "image_positions_prior": [
                    shifted["image"][0],
                    shifted["image"][-1],
                ],
                "question_positions_base": ordinary["question"],
                "question_positions_prior": shifted["question"],
                "option_positions_base": ordinary["options"],
                "option_positions_prior": shifted["options"],
                **control_metadata,
                "conditions": {
                    condition: {
                        "logits": logits.tolist(),
                        "prediction": "ABCD"[int(logits.argmax())],
                        "margin": margin(logits, int(gold[index])),
                    }
                    for condition, logits in logits_by_condition.items()
                },
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            del base_states
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == count:
                print(
                    f"hidden-shift {index + 1}/{count} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if module1_hashes() != before:
        raise RuntimeError("Module 1 source files changed during diagnosis")
    write_json(args.save_dir / "parity.json", parity)
    write_json(
        args.save_dir / "metadata.json",
        {
            "n": count,
            "conditions": list(CONDITIONS),
            "regions": list(REGIONS),
            "layers": 28,
            "metrics": list(SHIFT_NAMES),
            "module1_sha256": before,
            "seed": args.seed,
        },
    )
    reader.close()


if __name__ == "__main__":
    main()
