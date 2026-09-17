"""Frozen MedMO-8B reader with optional learned prefix tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image


MEDMO8B_DIR = Path(
    "/data/cyf/codes/YYY/VQA/MedMO/checkpoint/MedMO-8B-Next"
)
READER_KEY = "medmo_8b"
READER_NAME = "MedMO-8B-Next"
LABELS = "ABCD"


def option_values(row: dict[str, Any]) -> list[str]:
    options = row["options"]
    if isinstance(options, dict):
        values = [str(options[label]) for label in LABELS]
    else:
        values = [str(value) for value in options]
    if len(values) != 4:
        raise ValueError(f"Expected four options for {row['id']}")
    return values


def load_image(value: Any) -> Image.Image:
    import io

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


def medmo_prompt(
    question: str,
    options: list[str],
    prior_texts: list[str],
) -> str:
    option_lines = "\n".join(
        f"{label}. {value}" for label, value in zip(LABELS, options)
    )
    lines = ["Answer the visual medical multiple-choice question."]
    if prior_texts:
        lines.extend(
            [
                "Pediatric prior knowledge:",
                *[
                    f"[Pediatric Prior {index}]\n{text}"
                    for index, text in enumerate(prior_texts, start=1)
                ],
            ]
        )
    lines.extend(
        [
            f"Question: {question}",
            f"Options:\n{option_lines}",
            "Respond with exactly one character: A, B, C, or D.",
        ]
    )
    return "\n\n".join(lines)


class FrozenReader:
    def __init__(self, device_name: str):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = torch.device(device_name)
        self.processor = AutoProcessor.from_pretrained(
            str(MEDMO8B_DIR),
            local_files_only=True,
        )
        self.processor.tokenizer.padding_side = "left"
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MEDMO8B_DIR),
            local_files_only=True,
            dtype=dtype,
            device_map={"": str(self.device)}
            if self.device.type == "cuda"
            else None,
            attn_implementation="sdpa"
            if self.device.type == "cuda"
            else "eager",
        )
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model_device = next(self.model.parameters()).device
        config = self.model.config
        if hasattr(config, "hidden_size"):
            self.hidden_size = int(config.hidden_size)
        else:
            self.hidden_size = int(config.text_config.hidden_size)
        self.label_token_ids = self._label_token_ids()

    def _label_token_ids(self) -> list[int]:
        token_ids = []
        for label in LABELS:
            encoded = self.processor.tokenizer(
                label,
                add_special_tokens=False,
            )["input_ids"]
            if len(encoded) != 1:
                raise RuntimeError(f"{label} is not a single token: {encoded}")
            token_ids.append(int(encoded[0]))
        return token_ids

    def prompt(
        self,
        question: str,
        options: list[str],
        prior_texts: list[str],
    ) -> str:
        return medmo_prompt(question, options, prior_texts)

    def _prepare(
        self,
        rows: list[dict[str, Any]],
        tasks: list[tuple[int, list[str]]],
    ) -> tuple[dict[str, torch.Tensor], list[Image.Image]]:
        messages = []
        images = []
        for row_index, prior_texts in tasks:
            row = rows[row_index]
            image = load_image(row["image"])
            images.append(image)
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": self.prompt(
                                    str(row["question"]),
                                    option_values(row),
                                    prior_texts,
                                ),
                            },
                        ],
                    }
                ]
            )
        texts = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        moved = {
            key: value.to(self.model_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        return moved, images

    def logits(
        self,
        rows: list[dict[str, Any]],
        tasks: list[tuple[int, list[str]]],
        prefixes: torch.Tensor | None = None,
        require_grad: bool = False,
    ) -> torch.Tensor:
        inputs, images = self._prepare(rows, tasks)
        try:
            if prefixes is not None:
                if prefixes.ndim != 3 or prefixes.shape[0] != len(tasks):
                    raise ValueError(
                        f"Invalid prefix shape {tuple(prefixes.shape)} for "
                        f"{len(tasks)} tasks"
                    )
                input_ids = inputs["input_ids"]
                token_embeddings = self.model.get_input_embeddings()(input_ids)
                prefixes = prefixes.to(
                    self.model_device,
                    dtype=token_embeddings.dtype,
                )
                rope_kwargs = {
                    "input_ids": input_ids,
                    "image_grid_thw": inputs.get("image_grid_thw"),
                    "video_grid_thw": inputs.get("video_grid_thw"),
                    "attention_mask": inputs.get("attention_mask"),
                }
                position_ids, _ = self.model.model.get_rope_index(**rope_kwargs)
                prefix_length = prefixes.shape[1]
                prefix_position_ids = torch.arange(
                    prefix_length,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                ).view(1, 1, -1).expand(3, len(tasks), -1)
                inputs["position_ids"] = torch.cat(
                    [
                        prefix_position_ids,
                        position_ids + prefix_length,
                    ],
                    dim=-1,
                )
                inputs["inputs_embeds"] = torch.cat(
                    [prefixes, token_embeddings],
                    dim=1,
                )
                inputs.pop("input_ids")
                prefix_mask = torch.ones(
                    (len(tasks), prefixes.shape[1]),
                    dtype=inputs["attention_mask"].dtype,
                    device=self.model_device,
                )
                inputs["attention_mask"] = torch.cat(
                    [prefix_mask, inputs["attention_mask"]],
                    dim=1,
                )

            context = torch.enable_grad() if require_grad else torch.no_grad()
            with context:
                outputs = self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
                label_ids = torch.tensor(
                    self.label_token_ids,
                    dtype=torch.long,
                    device=outputs.logits.device,
                )
                return outputs.logits[:, -1, :].index_select(-1, label_ids)
        finally:
            for image in images:
                image.close()

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
