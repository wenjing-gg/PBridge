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
from reader import (
    FrozenReader,
    LABELS,
    MEDMO8B_DIR,
    READER_KEY,
    READER_NAME,
)


OUT = Path(__file__).resolve().parent / "artifacts"


def dataset_path(split: str) -> Path:
    return Path(
        "/nfsdata_a40/cyf/shared_data/pediatric_vqa/processed/"
        f"pediatrics_mqa_module1_8_2/{split}.parquet"
    )


def load_rows(split: str) -> list[dict[str, Any]]:
    return pq.read_table(dataset_path(split)).to_pylist()


def probabilities(
    reader: FrozenReader,
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
) -> dict[str, Any]:
    reader = FrozenReader(device_name)
    rows = load_rows(split)
    knowledge = read_jsonl(KNOWLEDGE_JSONL)
    retrieval_path = GME_OUT / f"{split}_retrieval.npz"
    with np.load(retrieval_path, allow_pickle=False) as archive:
        retrieval = {key: archive[key] for key in archive.files}
    candidate_indices = retrieval["candidate_indices"]
    if len(rows) != len(candidate_indices):
        raise RuntimeError("Reader rows and GME retrieval rows differ")

    no_rag_tasks = [(index, []) for index in range(len(rows))]
    no_rag = probabilities(
        reader,
        rows,
        no_rag_tasks,
        batch_size,
        f"{READER_NAME} {split} no-rag",
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
        f"{READER_NAME} {split} single-prior",
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

    reader_dir = OUT / READER_KEY
    reader_dir.mkdir(parents=True, exist_ok=True)
    output = reader_dir / f"{split}_assets.npz"
    np.savez_compressed(output, **output_values)
    metadata = {
        "reader": READER_NAME,
        "reader_model_dir": str(MEDMO8B_DIR),
        "split": split,
        "samples": len(rows),
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
    (reader_dir / f"{split}_assets.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    reader.close()
    return {"output": str(output), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.split, args.device, args.batch_size),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
