"""Record frozen no-prior and Top-3 outputs on the original training set."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from module2_alignment.common import (
    OUT,
    checkpoint_hash,
    frozen_ppr,
    split_ids,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()
    output = OUT / "counterfactual"
    output.mkdir(parents=True, exist_ok=True)
    reader, data, scores, ranks, prefixes, rows, gold = frozen_ppr(args.device)
    splits = split_ids(rows)
    write_json(
        output / "split.json",
        {
            "seed": 20260920,
            "train_indices": splits["train"].tolist(),
            "dev_indices": splits["dev"].tolist(),
            "checkpoint_sha256": checkpoint_hash(),
            "note": "Dev is held out of Module 2 fitting, not Module 1 fitting.",
        },
    )
    indices = np.arange(args.shard, len(rows), 2)
    started = time.monotonic()
    for begin in range(0, len(indices), args.chunk):
        subset = indices[begin : begin + args.chunk]
        path = output / f"top3_shard{args.shard}_{begin:04d}.npz"
        if path.is_file():
            with np.load(path) as existing:
                if np.array_equal(existing["index"], subset):
                    continue
            raise RuntimeError(f"Existing chunk indices mismatch: {path}")
        base = []
        prior = []
        for start in range(0, len(subset), args.batch_size):
            part = subset[start : start + args.batch_size]
            part_rows = [rows[int(i)] for i in part]
            tasks = [(i, []) for i in range(len(part))]
            with torch.no_grad():
                base.append(
                    reader.logits(part_rows, tasks)
                    .float().cpu().numpy()
                )
                prior.append(
                    reader.logits(
                        part_rows, tasks, prefixes=prefixes[part]
                    ).float().cpu().numpy()
                )
        np.savez_compressed(
            path,
            index=subset,
            gold=gold[subset],
            no_prior_logits=np.concatenate(base),
            top3_logits=np.concatenate(prior),
            ranks=ranks[subset],
            utility=data["utility_proxy"][subset],
            ppr_score=np.take_along_axis(scores[subset], ranks[subset], axis=1),
        )
        print(
            f"shard={args.shard} {min(begin + len(subset), len(indices))}/"
            f"{len(indices)} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    reader.close()


if __name__ == "__main__":
    main()
