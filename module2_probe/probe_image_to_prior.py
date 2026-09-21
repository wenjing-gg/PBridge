"""Extract image-query attention to the three PBridge prefix tokens."""

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
from module2_probe.mechanism_hooks import ImagePriorAttention
from module2_probe.probe_lingshu import (
    FrozenReader,
    answer_indices,
    load_ppr,
    load_rows,
    margin,
    spans,
)


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
        default=Path("module2_probe/outputs/mechanism/image_to_prior"),
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "blocks").mkdir(exist_ok=True)
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
                "input_length": ordinary["input_length"] + 3,
            }
            prefixes_by_condition, control_metadata = controls_for(index)
            base_logits = (
                reader.logits([row], [(0, [])])[0].float().cpu().numpy()
            )
            logits_by_condition = {}
            blocks = {}
            for condition, prefix in prefixes_by_condition.items():
                probe = ImagePriorAttention(shifted["image"])
                with torch.no_grad(), probe.active():
                    logits = reader.logits(
                        [row], [(0, [])], prefixes=prefix
                    )[0].float().cpu().numpy()
                logits_by_condition[condition] = logits
                blocks[condition] = probe.array()
                if blocks[condition].shape[:2] != (28, 28):
                    raise RuntimeError("Unexpected Lingshu attention shape")
            if index < 5:
                for condition, prefix in prefixes_by_condition.items():
                    native = reader.logits(
                        [row], [(0, [])], prefixes=prefix
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
                    if error > 1e-3 or not parity[-1]["same_prediction"]:
                        raise RuntimeError(
                            f"Attention probe changed logits: {parity[-1]}"
                        )
            np.savez_compressed(
                args.save_dir / "blocks" / f"{index:04d}.npz",
                **blocks,
            )
            record = {
                "index": index,
                "id": str(row["id"]),
                "gold": str(row["answer"]),
                "no_prior_logits": base_logits.tolist(),
                "no_prior_margin": margin(base_logits, int(gold[index])),
                "prior_ids": data["candidate_indices"][
                    index, ranks[index]
                ].tolist(),
                "prior_utility": data["utility_proxy"][
                    index, ranks[index]
                ].tolist(),
                "prior_score": scores[index, ranks[index]].tolist(),
                "image_positions": [
                    shifted["image"][0],
                    shifted["image"][-1],
                ],
                "question_positions": shifted["question"],
                "option_positions": shifted["options"],
                "image_grid_thw": ordinary["grid"],
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
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == count:
                print(
                    f"image-prior {index + 1}/{count} "
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
            "conditions": ["correct", "random", "shuffled"],
            "shape": "[layers=28, heads=28, image_tokens, prior_tokens=3]",
            "attention_backend": "original SDPA",
            "module1_sha256": before,
            "seed": args.seed,
        },
    )
    reader.close()


if __name__ == "__main__":
    main()
