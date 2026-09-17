"""Build the GME retrieval assets used by PBridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from retrieval import (
    dataset_path,
    encode_global_batch,
    gme_text,
    load_gme,
    load_image,
    resolve_device,
)


DATA_DIR = PROJECT_ROOT / "data" / "pediatric_knowledge"
KNOWLEDGE_JSONL = DATA_DIR / "knowledge.jsonl"
OUT = Path(__file__).resolve().parent / "artifacts"
KNOWLEDGE_EMBEDDINGS = OUT / "gme_knowledge_multimodal.npy"
MANIFEST = OUT / "gme_manifest.json"
TOP_K = 20


def resolve_knowledge_image(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def encode_knowledge(device_name: str, batch_size: int) -> dict[str, Any]:
    knowledge = read_jsonl(KNOWLEDGE_JSONL)
    device = resolve_device(device_name)
    model, processor = load_gme(device)
    output = np.empty((len(knowledge), 1536), dtype=np.float16)

    groups = {
        "text_only": [
            index
            for index, item in enumerate(knowledge)
            if not item.get("image_paths")
        ],
        "image_text": [
            index
            for index, item in enumerate(knowledge)
            if item.get("image_paths")
        ],
    }
    for modality, indices in groups.items():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = [knowledge[index] for index in batch_indices]
            images = None
            if modality == "image_text":
                images = [
                    load_image(resolve_knowledge_image(item["image_paths"][0]))
                    for item in batch
                ]
            embeddings = encode_global_batch(
                model,
                processor,
                [
                    gme_text(
                        str(item.get("knowledge_text") or item["text"]),
                        modality == "image_text",
                    )
                    for item in batch
                ],
                images,
                device,
            )
            output[batch_indices] = embeddings
            if images is not None:
                for image in images:
                    image.close()
            print(
                f"{modality}: {min(start + batch_size, len(indices))}/"
                f"{len(indices)}",
                flush=True,
            )

    np.save(KNOWLEDGE_EMBEDDINGS, output)
    write_json(
        MANIFEST,
        {
            "model_id": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            "knowledge_items": len(knowledge),
            "embedding_shape": list(output.shape),
            "text_only_items": len(groups["text_only"]),
            "image_text_items": len(groups["image_text"]),
            "image_policy": "first image_paths entry for Pediatric Imaging items",
            "all_frozen": True,
        },
    )
    del model
    return {
        "output": str(KNOWLEDGE_EMBEDDINGS),
        "items": len(knowledge),
        "shape": list(output.shape),
    }


def encode_queries(
    rows: list[dict[str, Any]],
    device_name: str,
    batch_size: int,
) -> np.ndarray:
    device = resolve_device(device_name)
    model, processor = load_gme(device)
    output = np.empty((len(rows), 1536), dtype=np.float16)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [load_image(row["image"]) for row in batch]
        output[start : start + len(batch)] = encode_global_batch(
            model,
            processor,
            [gme_text(str(row["question"]), True) for row in batch],
            images,
            device,
        )
        for image in images:
            image.close()
        print(
            f"queries: {min(start + batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )
    del model
    return output


def top_k_rows(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
    local_scores = np.take_along_axis(scores, local, axis=1)
    order = np.argsort(-local_scores, axis=1, kind="stable")
    return (
        np.take_along_axis(local, order, axis=1),
        np.take_along_axis(local_scores, order, axis=1),
    )


def build_candidates(
    split: str,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    dataset = dataset_path(split)
    rows = pq.read_table(
        dataset,
        columns=["id", "question", "image"],
    ).to_pylist()
    query = encode_queries(rows, device_name, batch_size).astype(np.float32)
    knowledge = np.asarray(
        np.load(KNOWLEDGE_EMBEDDINGS, mmap_mode="r"),
        dtype=np.float32,
    )
    scores = query @ knowledge.T
    indices, top_scores = top_k_rows(scores, TOP_K)
    output = OUT / f"{split}_retrieval.npz"
    np.savez_compressed(
        output,
        sample_ids=np.asarray([str(row["id"]) for row in rows]),
        query_embeddings=query.astype(np.float32),
        candidate_indices=indices.astype(np.int32),
        candidate_embeddings=knowledge[indices].astype(np.float16),
        gme_scores=top_scores.astype(np.float32),
    )
    return {
        "split": split,
        "samples": len(rows),
        "output": str(output),
        "candidates": TOP_K,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["encode-knowledge", "build-candidates", "all"],
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=["train", "test"])
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.command in {"encode-knowledge", "all"}:
        print(encode_knowledge(args.device, args.batch_size), flush=True)
    if args.command in {"build-candidates", "all"}:
        if not KNOWLEDGE_EMBEDDINGS.is_file():
            raise FileNotFoundError(
                f"Missing {KNOWLEDGE_EMBEDDINGS}; run encode-knowledge first"
            )
        splits = [args.split] if args.split else ["train", "test"]
        for split in splits:
            print(
                build_candidates(split, args.device, args.batch_size),
                flush=True,
            )


if __name__ == "__main__":
    main()
