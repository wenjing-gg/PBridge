#!/usr/bin/env python3
"""Create the fixed image-grouped 80:20 PediatricsMQA split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from retrieval import (
    SOURCE_DATASET,
    SPLIT_DIR,
    SPLIT_MANIFEST,
    TEST_DATASET,
    TRAIN_DATASET,
)


SEED = 20260913
TRAIN_FRACTION = 0.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_test_groups(
    rows: list[dict[str, object]],
    target_test_rows: int,
) -> set[str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["img_id"])].append(index)
    rng = np.random.default_rng(SEED)
    group_ids = list(groups)
    rng.shuffle(group_ids)

    choices: dict[int, tuple[str, ...]] = {0: ()}
    for group_id in group_ids:
        size = len(groups[group_id])
        for total, selected in list(choices.items())[::-1]:
            new_total = total + size
            if new_total <= target_test_rows and new_total not in choices:
                choices[new_total] = selected + (group_id,)
    if target_test_rows not in choices:
        raise RuntimeError(
            f"Could not form an exact {target_test_rows}-row test split"
        )
    return set(choices[target_test_rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DATASET)
    args = parser.parse_args()
    source = args.source
    if not source.is_file():
        raise FileNotFoundError(str(source))

    table = pq.read_table(source)
    rows = table.select(["id", "img_id"]).to_pylist()
    total_rows = len(rows)
    target_train_rows = round(total_rows * TRAIN_FRACTION)
    target_test_rows = total_rows - target_train_rows
    test_groups = choose_test_groups(rows, target_test_rows)
    test_indices = [
        index for index, row in enumerate(rows)
        if str(row["img_id"]) in test_groups
    ]
    train_indices = [
        index for index, row in enumerate(rows)
        if str(row["img_id"]) not in test_groups
    ]
    if len(train_indices) != target_train_rows:
        raise RuntimeError("Train split does not match the requested 80:20 size")
    if set(rows[index]["img_id"] for index in train_indices) & set(
        rows[index]["img_id"] for index in test_indices
    ):
        raise RuntimeError("An img_id crosses the train/test boundary")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(train_indices)), TRAIN_DATASET)
    pq.write_table(table.take(pa.array(test_indices)), TEST_DATASET)
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source),
        "seed": SEED,
        "group_key": "img_id",
        "train_fraction": TRAIN_FRACTION,
        "test_fraction": 1.0 - TRAIN_FRACTION,
        "total_rows": total_rows,
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "total_img_groups": len(set(str(row["img_id"]) for row in rows)),
        "train_img_groups": len(
            set(str(rows[index]["img_id"]) for index in train_indices)
        ),
        "test_img_groups": len(
            set(str(rows[index]["img_id"]) for index in test_indices)
        ),
        "train_dataset": str(TRAIN_DATASET),
        "test_dataset": str(TEST_DATASET),
    }
    SPLIT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
