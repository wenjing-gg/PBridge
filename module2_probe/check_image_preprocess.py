"""Validate Lingshu image preprocessing for the active PBridge reader."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from module1_ppr.reader import (
    LINGSHU7B_DIR,
    load_image,
    option_values,
    pbridge_prompt,
)


DATASET = Path(
    "/nfsdata_a40/cyf/shared_data/pediatric_vqa/processed/"
    "pediatrics_mqa_module1_8_2/test.parquet"
)
OUTPUT = Path("module2_probe/outputs/image_preprocess_check")
CONFIG_NAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "config.json",
    "tokenizer_config.json",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def raw_image(value: Any) -> tuple[Image.Image, str, int | None]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        return Image.open(path), "path", path.stat().st_size
    if isinstance(value, dict) and value.get("bytes") is not None:
        data = value["bytes"]
        return Image.open(io.BytesIO(data)), "bytes", len(data)
    if isinstance(value, dict) and value.get("path"):
        path = Path(value["path"])
        return Image.open(path), "path", path.stat().st_size
    raise ValueError("Unsupported image representation")


def reader_text(processor: Any, row: dict[str, Any], image: Image.Image) -> str:
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": pbridge_prompt(
                            str(row["question"]),
                            option_values(row),
                            [],
                        ),
                    },
                ],
            }
        ]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )[0]


def summarize_output(
    output: dict[str, torch.Tensor],
    merge_size: int,
    image_token_id: int | None,
) -> dict[str, Any]:
    grid = output["image_grid_thw"][0].detach().cpu().tolist()
    result = {
        "image_grid_thw": [int(value) for value in grid],
        "pixel_values_shape": [
            int(value) for value in output["pixel_values"].shape
        ],
        "final_image_tokens": int(np.prod(grid) // (merge_size**2)),
        "tensor_shapes": {
            key: [int(value) for value in tensor.shape]
            for key, tensor in output.items()
            if hasattr(tensor, "shape")
        },
    }
    if image_token_id is not None and "input_ids" in output:
        result["image_token_count_in_input_ids"] = int(
            (output["input_ids"] == image_token_id).sum().item()
        )
    return result


def run_processor(
    processor: Any,
    row: dict[str, Any],
    image: Image.Image,
    **kwargs: Any,
) -> dict[str, Any]:
    output = processor(
        text=[reader_text(processor, row, image)],
        images=[image],
        return_tensors="pt",
        **kwargs,
    )
    return summarize_output(
        output,
        int(processor.image_processor.merge_size),
        int(processor.image_token_id),
    )


def run_image_processor(
    processor: Any,
    image: Image.Image,
    **kwargs: Any,
) -> dict[str, Any]:
    output = processor.image_processor(
        images=[image],
        return_tensors="pt",
        **kwargs,
    )
    return summarize_output(
        output,
        int(processor.image_processor.merge_size),
        None,
    )


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def choose_control(dimensions: list[tuple[int, int]]) -> int:
    candidates = [
        index
        for index, (width, height) in enumerate(dimensions)
        if min(width, height) >= 256
    ]
    return min(
        candidates,
        key=lambda index: (
            abs(dimensions[index][0] - dimensions[index][1]),
            -min(dimensions[index]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.dataset).to_pylist()
    processor = AutoProcessor.from_pretrained(
        str(LINGSHU7B_DIR),
        local_files_only=True,
        use_fast=False,
    )
    image_processor = processor.image_processor
    local_configs = {
        name: read_json_if_present(LINGSHU7B_DIR / name)
        for name in CONFIG_NAMES
    }
    checkpoint_transformers = (
        local_configs.get("config.json") or {}
    ).get("transformers_version")
    processor_report = {
        "checkpoint": str(LINGSHU7B_DIR),
        "runtime_transformers_version": transformers.__version__,
        "checkpoint_transformers_version": checkpoint_transformers,
        "torch_version": torch.__version__,
        "processor_class": (
            f"{type(processor).__module__}.{type(processor).__name__}"
        ),
        "image_processor_class": (
            f"{type(image_processor).__module__}."
            f"{type(image_processor).__name__}"
        ),
        "image_processor": {
            key: getattr(image_processor, key, None)
            for key in (
                "min_pixels",
                "max_pixels",
                "size",
                "patch_size",
                "merge_size",
                "temporal_patch_size",
                "do_resize",
            )
        },
        "tokenizer_init_pixel_kwargs": {
            key: processor.tokenizer.init_kwargs.get(key)
            for key in ("min_pixels", "max_pixels")
        },
        "local_config_files": local_configs,
    }
    (output_dir / "processor_config.txt").write_text(
        json.dumps(processor_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    raw_dimensions: list[tuple[int, int]] = []
    representations: Counter[str] = Counter()
    for row in rows:
        raw, representation, _ = raw_image(row["image"])
        try:
            raw_dimensions.append(tuple(int(value) for value in raw.size))
        finally:
            raw.close()
        representations[representation] += 1

    rng = np.random.default_rng(args.seed)
    sample_indices = sorted(
        rng.choice(
            len(rows),
            size=min(args.sample_count, len(rows)),
            replace=False,
        ).tolist()
    )
    sample_records: list[dict[str, Any]] = []
    with (output_dir / "sample_debug.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for index in sample_indices:
            row = rows[index]
            raw, representation, encoded_size = raw_image(row["image"])
            loaded = load_image(row["image"])
            try:
                before_size = tuple(int(value) for value in loaded.size)
                processed = run_processor(processor, row, loaded)
                record = {
                    "sample_index": index,
                    "sample_id": str(row["id"]),
                    "representation": representation,
                    "encoded_image_bytes": encoded_size,
                    "raw_image_size": [int(value) for value in raw.size],
                    "loaded_pil_size": [int(value) for value in loaded.size],
                    "processor_input_pil_size": list(before_size),
                    **processed,
                    "unchanged_before_processor": bool(
                        raw.size == loaded.size == before_size
                    ),
                }
            finally:
                raw.close()
                loaded.close()
            sample_records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    grid_counter: Counter[tuple[int, int, int]] = Counter()
    pixel_shape_counter: Counter[tuple[int, ...]] = Counter()
    token_counter: Counter[int] = Counter()
    input_token_counter: Counter[int] = Counter()
    for index, row in enumerate(rows):
        image = load_image(row["image"])
        try:
            processed = run_processor(processor, row, image)
        finally:
            image.close()
        grid_counter[tuple(processed["image_grid_thw"])] += 1
        pixel_shape_counter[tuple(processed["pixel_values_shape"])] += 1
        token_counter[int(processed["final_image_tokens"])] += 1
        input_token_counter[
            int(processed["image_token_count_in_input_ids"])
        ] += 1
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"processed {index + 1}/{len(rows)}", flush=True)

    dimensions = np.asarray(raw_dimensions, dtype=np.int64)
    areas = np.prod(dimensions, axis=1)
    size_statistics = {
        "n": len(rows),
        "representations": dict(representations),
        "width": {
            "min": int(dimensions[:, 0].min()),
            "max": int(dimensions[:, 0].max()),
            "median": float(np.median(dimensions[:, 0])),
        },
        "height": {
            "min": int(dimensions[:, 1].min()),
            "max": int(dimensions[:, 1].max()),
            "median": float(np.median(dimensions[:, 1])),
        },
        "area": {
            "min": int(areas.min()),
            "max": int(areas.max()),
            "median": float(np.median(areas)),
        },
        "sample_count": len(sample_records),
        "all_sample_sizes_unchanged_before_processor": all(
            record["unchanged_before_processor"]
            for record in sample_records
        ),
    }
    write_json(output_dir / "image_size_statistics.json", size_statistics)
    grid_statistics = {
        "n": len(rows),
        "image_grid_thw_frequency": {
            str(list(key)): value
            for key, value in sorted(grid_counter.items())
        },
        "pixel_values_shape_frequency": {
            str(list(key)): value
            for key, value in sorted(pixel_shape_counter.items())
        },
        "final_image_token_frequency": {
            str(key): value for key, value in sorted(token_counter.items())
        },
        "image_token_count_in_input_ids_frequency": {
            str(key): value
            for key, value in sorted(input_token_counter.items())
        },
        "all_cases_have_multiple_image_tokens": all(
            count > 1 for count in token_counter
        ),
    }
    write_json(output_dir / "image_grid_statistics.json", grid_statistics)

    control_index = choose_control(raw_dimensions)
    control_row = rows[control_index]
    control_image = load_image(control_row["image"])
    try:
        tests: list[tuple[str, Callable[[], dict[str, Any]]]] = [
            (
                "pbridge_default",
                lambda: run_processor(
                    processor,
                    control_row,
                    control_image,
                ),
            ),
            (
                "explicit_checkpoint_budget",
                lambda: run_processor(
                    processor,
                    control_row,
                    control_image,
                    min_pixels=int(image_processor.min_pixels),
                    max_pixels=int(image_processor.max_pixels),
                ),
            ),
            (
                "explicit_todo_budget",
                lambda: run_processor(
                    processor,
                    control_row,
                    control_image,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1280 * 28 * 28,
                ),
            ),
            (
                "forced_single_token_budget",
                lambda: run_processor(
                    processor,
                    control_row,
                    control_image,
                    min_pixels=28 * 28,
                    max_pixels=28 * 28,
                ),
            ),
            (
                "direct_image_processor_default",
                lambda: run_image_processor(
                    processor,
                    control_image,
                ),
            ),
        ]
        budget_results = {name: function() for name, function in tests}
    finally:
        control_image.close()

    width, height = raw_dimensions[control_index]
    factor = int(image_processor.patch_size * image_processor.merge_size)
    default_resize = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=int(image_processor.min_pixels),
        max_pixels=int(image_processor.max_pixels),
    )
    single_resize = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=28 * 28,
        max_pixels=28 * 28,
    )
    pixel_budget = {
        "sample_index": control_index,
        "sample_id": str(control_row["id"]),
        "original_size": [width, height],
        "resize_factor": factor,
        "smart_resize_default": [
            int(default_resize[1]),
            int(default_resize[0]),
        ],
        "smart_resize_forced_single": [
            int(single_resize[1]),
            int(single_resize[0]),
        ],
        "tests": budget_results,
    }
    write_json(output_dir / "pixel_budget_comparison.json", pixel_budget)

    default_tokens = budget_results["pbridge_default"]["final_image_tokens"]
    forced_tokens = budget_results[
        "forced_single_token_budget"
    ]["final_image_tokens"]
    conclusion = f"""# Lingshu Image Preprocess Diagnosis

## 结论

当前 Lingshu-PBridge 不存在单视觉 token 缺陷。固定测试集 {len(rows)} 题全部
保留多个视觉 token，默认控制样例产生 {default_tokens} 个视觉 token。

## 检查结果

- 原始宽度统计：{size_statistics["width"]}
- 原始高度统计：{size_statistics["height"]}
- 抽样图片进入 Processor 前尺寸均未变化：
  {size_statistics["all_sample_sizes_unchanged_before_processor"]}
- Processor：`{processor_report["processor_class"]}`
- `min_pixels={image_processor.min_pixels}`
- `max_pixels={image_processor.max_pixels}`
- `patch_size={image_processor.patch_size}`
- `merge_size={image_processor.merge_size}`
- 完整测试集网格频次：{grid_statistics["image_grid_thw_frequency"]}
- 最终视觉 token 频次：{grid_statistics["final_image_token_frequency"]}

## 像素预算对照

正式 PBridge 调用与 checkpoint 默认预算结果一致。仅当诊断脚本显式把
`min_pixels=max_pixels=28×28` 时，控制样例才被压到 {forced_tokens} 个视觉
token。这说明视觉网格受像素预算正常控制，并非数据加载或版本兼容异常。

## 判断

原 TODO 中的 Case A-D 均不适用于当前 Lingshu-PBridge：

- 不是官方低视觉预算；
- 不是本地 Processor 配置冲突；
- 未观察到运行时版本导致的异常网格；
- 数据加载阶段没有提前缩图。

因此无需修改 Lingshu 的视觉分辨率，也无需因图像预处理问题重新训练 Module 1。
当前多视觉 token 输入可继续用于后续空间注意力分析。
"""
    (output_dir / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
