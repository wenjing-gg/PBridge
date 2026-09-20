"""Compute reader-specific counterfactual utility for PBridge PPR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from gme_assets import KNOWLEDGE_JSONL, OUT as GME_OUT, read_jsonl
from reader import FrozenReader, LABELS, READER_KEY


OUT = Path(__file__).resolve().parent / "artifacts"


def dataset_path(split: str) -> Path:
    return Path(
        "/nfsdata_a40/cyf/shared_data/pediatric_vqa/processed/"
        f"pediatrics_mqa_module1_8_2/{split}.parquet"
    )


def load_rows(split: str) -> list[dict[str, Any]]:
    return pq.read_table(dataset_path(split)).to_pylist()


def probabilities(
    reader: Any,
    rows: list[dict[str, Any]],
    tasks: list[tuple[int, list[str]]],
    batch_size: int,
    label: str,
) -> np.ndarray:
    output = np.empty((len(tasks), 4), dtype=np.float32)
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        logits = reader.logits(rows, batch)
        output[start : start + len(batch)] = (
            torch.softmax(logits.float(), dim=-1).cpu().numpy()
        )
        end = start + len(batch)
        if start == 0 or end == len(tasks) or end % 500 == 0:
            print(f"{label}: {end}/{len(tasks)}", flush=True)
    return output


def margins(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(values, 1e-12, 1.0))
    flat_labels = labels.reshape(-1)
    flat = logp.reshape(-1, 4)
    selected = flat[np.arange(len(flat)), flat_labels]
    alternatives = flat.copy()
    alternatives[np.arange(len(flat)), flat_labels] = -np.inf
    result = selected - alternatives.max(axis=1)
    return result.reshape(labels.shape)


def prepare(
    split: str,
    device_name: str,
    batch_size: int,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"Invalid shard {shard_index} for num_shards={num_shards}"
        )
    all_rows = load_rows(split)
    knowledge = read_jsonl(KNOWLEDGE_JSONL)
    retrieval_path = GME_OUT / f"{split}_retrieval.npz"
    with np.load(retrieval_path, allow_pickle=False) as archive:
        retrieval = {key: archive[key] for key in archive.files}
    if len(all_rows) != len(retrieval["candidate_indices"]):
        raise RuntimeError("Reader rows and GME retrieval rows differ")
    boundaries = np.linspace(
        0,
        len(all_rows),
        num_shards + 1,
        dtype=np.int64,
    )
    row_start = int(boundaries[shard_index])
    row_end = int(boundaries[shard_index + 1])
    rows = all_rows[row_start:row_end]
    retrieval = {
        key: value[row_start:row_end]
        for key, value in retrieval.items()
    }
    candidate_indices = retrieval["candidate_indices"]
    reader = FrozenReader(device_name)

    no_rag_tasks = [(index, []) for index in range(len(rows))]
    no_rag = probabilities(
        reader,
        rows,
        no_rag_tasks,
        batch_size,
        f"{reader.reader_name} {split} no-rag",
    )
    single_tasks = [
        (
            row_index,
            [str(knowledge[int(candidate)]["knowledge_text"])],
        )
        for row_index in range(len(rows))
        for candidate in candidate_indices[row_index]
    ]
    single = probabilities(
        reader,
        rows,
        single_tasks,
        batch_size,
        f"{reader.reader_name} {split} single-prior",
    ).reshape(len(rows), candidate_indices.shape[1], 4)

    predicted = no_rag.argmax(axis=1)
    predicted_grid = np.broadcast_to(
        predicted[:, None],
        candidate_indices.shape,
    )
    utility_proxy = margins(single, predicted_grid) - margins(
        no_rag,
        predicted,
    )[:, None]

    output_values: dict[str, np.ndarray] = {
        "query_embeddings": retrieval["query_embeddings"],
        "candidate_embeddings": retrieval["candidate_embeddings"],
        "utility_proxy": utility_proxy.astype(np.float32),
        "no_rag_probabilities": no_rag,
    }
    if split == "test":
        output_values["candidate_indices"] = candidate_indices
    if split == "train":
        gold = np.asarray(
            [LABELS.index(str(row["answer"]).strip().upper()) for row in rows],
            dtype=np.int64,
        )
        gold_grid = np.broadcast_to(gold[:, None], candidate_indices.shape)
        utility_target = margins(single, gold_grid) - margins(
            no_rag,
            gold,
        )[:, None]
        output_values["utility_target"] = utility_target.astype(np.float32)
        output_values["gold"] = gold

    reader_dir = OUT / reader.reader_key
    reader_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        ""
        if num_shards == 1
        else f".shard-{shard_index:03d}-of-{num_shards:03d}"
    )
    output = reader_dir / f"{split}_assets{suffix}.npz"
    np.savez_compressed(output, **output_values)
    metadata = {
        "reader": reader.reader_name,
        "reader_model_dir": str(reader.model_dir),
        "visual_token_mode": reader.visual_token_mode,
        "image_min_pixels": reader.image_min_pixels,
        "image_max_pixels": reader.image_max_pixels,
        "split": split,
        "samples": len(rows),
        "row_start": row_start,
        "row_end": row_end,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "candidates": int(candidate_indices.shape[1]),
        "utility_target": (
            "gold-conditioned train-only counterfactual margin"
            if split == "train"
            else "not computed"
        ),
        "utility_proxy": (
            "counterfactual margin for the frozen reader's no-RAG prediction; "
            "does not use the gold answer"
        ),
        "single_prior_protocol": "one raw knowledge_text in the reader prompt",
    }
    (reader_dir / f"{split}_assets{suffix}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    reader.close()
    return {"output": str(output), **metadata}


def merge_shards(
    split: str,
    num_shards: int,
) -> dict[str, Any]:
    reader_dir = OUT / READER_KEY
    shard_paths = [
        reader_dir
        / f"{split}_assets.shard-{index:03d}-of-{num_shards:03d}.npz"
        for index in range(num_shards)
    ]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing}")
    shards = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as archive:
            shards.append({key: archive[key] for key in archive.files})
    keys = set(shards[0])
    if any(set(shard) != keys for shard in shards[1:]):
        raise RuntimeError("Reader asset shards have inconsistent keys")
    merged = {
        key: np.concatenate([shard[key] for shard in shards], axis=0)
        for key in sorted(keys)
    }
    expected = len(load_rows(split))
    if any(len(value) != expected for value in merged.values()):
        sizes = {key: len(value) for key, value in merged.items()}
        raise RuntimeError(
            f"Merged assets do not match {expected} rows: {sizes}"
        )
    output = reader_dir / f"{split}_assets.npz"
    np.savez_compressed(output, **merged)
    metadata = {
        "reader": "Lingshu-7B",
        "reader_model_dir": "/data/cyf/codes/YYY/VQA/Lingshu-7B/checkpoint",
        "visual_token_mode": "native_multiple",
        "image_min_pixels": 3136,
        "image_max_pixels": 12845056,
        "reader_key": READER_KEY,
        "split": split,
        "samples": expected,
        "num_shards": num_shards,
        "output": str(output),
    }
    (reader_dir / f"{split}_assets.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        result = merge_shards(args.split, args.num_shards)
    else:
        result = prepare(
            args.split,
            args.device,
            args.batch_size,
            args.shard_index,
            args.num_shards,
        )
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
