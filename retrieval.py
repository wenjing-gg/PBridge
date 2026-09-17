"""Shared GME encoder utilities for PBridge."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
GME_MODEL_DIR = ROOT / "models" / "gme"
GME_INSTRUCTION = (
    "Retrieve pediatric medical knowledge relevant to the image and "
    "clinical question."
)
SOURCE_DATASET = Path(
    "/nfsdata_a40/cyf/shared_data/pediatric_vqa/raw/pediatric/"
    "PediatricsMQA/vqa/test-00000-of-00001.parquet"
)
SPLIT_DIR = Path(
    "/nfsdata_a40/cyf/shared_data/pediatric_vqa/processed/"
    "pediatrics_mqa_module1_8_2"
)
TRAIN_DATASET = SPLIT_DIR / "train.parquet"
TEST_DATASET = SPLIT_DIR / "test.parquet"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.json"

MIN_IMAGE_TOKENS = 256
MAX_IMAGE_TOKENS = 1280
MAX_SEQUENCE_LENGTH = 1800


def dataset_path(split: str) -> Path:
    path = {"train": TRAIN_DATASET, "test": TEST_DATASET}.get(split)
    if path is None:
        raise ValueError(f"Unknown dataset split: {split}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split} split at {path}; run "
            "prepare_split.py first"
        )
    return path


def resolve_device(device: str | None) -> torch.device:
    requested = device or os.environ.get("PEDIA_VQA_DEVICE", "cuda:0")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
    return torch.device(requested)


def required_model_files() -> list[Path]:
    return [
        GME_MODEL_DIR / "config.json",
        GME_MODEL_DIR / "model.safetensors.index.json",
        GME_MODEL_DIR / "model-00001-of-00003.safetensors",
        GME_MODEL_DIR / "model-00002-of-00003.safetensors",
        GME_MODEL_DIR / "model-00003-of-00003.safetensors",
        GME_MODEL_DIR / "preprocessor_config.json",
        GME_MODEL_DIR / "tokenizer.json",
        GME_MODEL_DIR / "tokenizer_config.json",
    ]


def load_gme(device: torch.device) -> tuple[Any, Any]:
    missing = [str(path) for path in required_model_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "GME checkpoint is incomplete; missing local files: " + str(missing)
        )
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": str(device)}
        load_kwargs["attn_implementation"] = "sdpa"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(GME_MODEL_DIR),
        **load_kwargs,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(GME_MODEL_DIR),
        local_files_only=True,
        use_fast=False,
        min_pixels=MIN_IMAGE_TOKENS * 28 * 28,
        max_pixels=MAX_IMAGE_TOKENS * 28 * 28,
    )
    processor.tokenizer.padding_side = "right"
    return model, processor


def load_image(value: Any) -> Image.Image:
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB").copy()
    if isinstance(value, dict) and value.get("bytes") is not None:
        with Image.open(io.BytesIO(value["bytes"])) as image:
            return image.convert("RGB").copy()
    if isinstance(value, dict) and value.get("path"):
        with Image.open(value["path"]) as image:
            return image.convert("RGB").copy()
    raise ValueError("Unsupported image representation")


def gme_text(text: str, with_image: bool) -> str:
    image_tokens = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        if with_image
        else ""
    )
    return (
        f"<|im_start|>system\n{GME_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{image_tokens}{text}<|im_end|>\n"
        "<|im_start|>assistant\n<|endoftext|>"
    )


def move_inputs(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(values.float(), p=2, dim=-1)


def encode_global_batch(
    model: Any,
    processor: Any,
    texts: list[str],
    images: list[Image.Image] | None,
    device: torch.device,
) -> np.ndarray:
    kwargs: dict[str, Any] = {
        "text": texts,
        "padding": True,
        "truncation": True,
        "max_length": MAX_SEQUENCE_LENGTH,
        "return_tensors": "pt",
    }
    if images is not None:
        kwargs["images"] = images
    inputs = move_inputs(processor(**kwargs), device)
    allowed = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    }
    model_inputs = {
        key: value for key, value in inputs.items() if key in allowed
    }
    with torch.inference_mode():
        outputs = model.model(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    lengths = inputs["attention_mask"].sum(dim=1) - 1
    positions = lengths.to(outputs.last_hidden_state.device)
    batch_indices = torch.arange(
        len(texts),
        device=outputs.last_hidden_state.device,
    )
    embeddings = normalize_rows(
        outputs.last_hidden_state[batch_indices, positions]
    )
    return embeddings.cpu().numpy().astype(np.float16)
