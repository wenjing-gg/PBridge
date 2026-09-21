"""Paired Lingshu-PBridge attention probing without edits to Module 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from module2_probe.metrics import spatial_metrics
from module2_probe.fixed_module1 import module1_hashes
from module2_probe.probe_lingshu import (
    FrozenReader, ROOT, answer_indices, load_assets, load_ppr, load_rows,
    margin, run, spans,
)
from module1_ppr.gme_assets import OUT as GME_OUT
from module1_ppr.model import PPRMoE


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "module1_ppr/artifacts/lingshu_7b/ppr_moe.pt")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--mode", choices=("both",), default="both")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "module2_probe/outputs/attention_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "blocks").mkdir(exist_ok=True)
    before = module1_hashes()
    reader = FrozenReader(args.device)
    data, scores, ranks, prefixes = load_ppr(reader, args.checkpoint, args.split)
    rows = load_rows(args.split)
    gold = answer_indices(rows)
    if len(rows) != len(prefixes):
        raise RuntimeError("Dataset/reader asset mismatch")
    if args.split == "test":
        candidate_indices = data["candidate_indices"]
    else:
        with np.load(GME_OUT / "train_retrieval.npz") as archive:
            candidate_indices = archive["candidate_indices"]
    from module1_ppr.gme_assets import KNOWLEDGE_EMBEDDINGS
    bank = np.load(KNOWLEDGE_EMBEDDINGS, mmap_mode="r")
    model = PPRMoE(reader.hidden_size).to(reader.model_device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    count = len(rows) if args.max_cases <= 0 else min(len(rows), args.max_cases)
    indices = np.sort(rng.choice(len(rows), size=count, replace=False))
    architecture = {
        "layers": len(reader.model.model.language_model.layers),
        "heads": reader.model.config.text_config.num_attention_heads,
        "key_value_heads": reader.model.config.text_config.num_key_value_heads,
        "hidden_size": reader.hidden_size,
        "head_dim": reader.hidden_size // reader.model.config.text_config.num_attention_heads,
        "attention_backend": reader.model.config.text_config._attn_implementation,
        "vision_layers": len(reader.model.model.visual.blocks),
        "spatial_merge_size": reader.processor.image_processor.merge_size,
        "checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "module1_sha256": before,
        "split": args.split, "count": count, "seed": args.seed,
        "conditional_spatial_metrics": True,
    }
    write_json(args.save_dir / "architecture.json", architecture)
    parity = []
    started = time.monotonic()
    with (args.save_dir / "cases.jsonl").open("w") as output:
        for n, idx in enumerate(indices):
            idx = int(idx)
            row = rows[idx]
            ordinary = spans(reader, row)
            shifted = {key: (
                [value + 3 for value in ordinary[key]]
                if key in ("image", "question", "options")
                else value + 3 if key in ("last", "input_length") else value
            ) for key, value in ordinary.items()}
            pair = prefixes[idx : idx + 1]
            if pair.shape != (1, 3, reader.hidden_size):
                raise RuntimeError("PPR prefix has unexpected shape")
            base_logits, base_attention = run(reader, row, None, ordinary)
            prior_logits, prior_attention = run(reader, row, pair, shifted)
            if base_attention["q_image"].shape != prior_attention["q_image"].shape:
                raise RuntimeError("Prior changed the image/question span")
            if n < 5:
                with torch.no_grad():
                    native_base = reader.logits([row], [(0, [])])[0].float().cpu().numpy()
                    native_prior = reader.logits(
                        [row], [(0, [])], prefixes=pair
                    )[0].float().cpu().numpy()
                error = max(
                    float(np.max(abs(native_base - base_logits))),
                    float(np.max(abs(native_prior - prior_logits))),
                )
                parity.append({"index": idx, "max_abs_logit_error": error,
                               "same_prediction": bool(
                                   native_base.argmax() == base_logits.argmax()
                                   and native_prior.argmax() == prior_logits.argmax()),
                               "max_abs_margin_error": max(
                                   abs(margin(native_base, int(gold[idx])) - margin(base_logits, int(gold[idx]))),
                                   abs(margin(native_prior, int(gold[idx])) - margin(prior_logits, int(gold[idx]))),
                               )})
                if error > 1e-3 or not parity[-1]["same_prediction"]:
                    raise RuntimeError(f"Probe changes the native reader: {parity[-1]}")
            metrics = spatial_metrics(base_attention["q_image"], prior_attention["q_image"])
            top_ranks = ranks[idx]
            selected_utility = data["utility_proxy"][idx, top_ranks]
            # Controls keep the same image/question and the same three prefix slots.
            donor_idx = int((idx + 1 + rng.integers(len(rows) - 1)) % len(rows))
            random_indices = rng.choice(len(bank), size=3, replace=False)
            with torch.no_grad():
                random_knowledge = torch.as_tensor(
                    np.asarray(bank[random_indices], dtype=np.float32),
                    device=reader.model_device,
                )[None]
                query = torch.as_tensor(
                    data["query_embeddings"][idx],
                    device=reader.model_device,
                )[None]
                proxy = torch.as_tensor(
                    selected_utility,
                    device=reader.model_device,
                )[None]
                random_scores, _, _ = model(
                    query, random_knowledge, proxy
                )
                random_ranks = random_scores.topk(3, dim=-1).indices
                randomized = model.prefix_embeddings(
                    random_knowledge,
                    random_scores,
                    random_ranks,
                )
            controls = {
                "random": randomized, "shuffled": prefixes[donor_idx : donor_idx + 1],
                "zero": torch.zeros_like(pair),
            }
            comparisons = {}
            for name, control_prefix in controls.items():
                control_logits, control_attention = run(reader, row, control_prefix, shifted)
                control_metrics = spatial_metrics(
                    base_attention["q_image"], control_attention["q_image"]
                )
                comparisons[name] = {
                    "logits": control_logits.tolist(),
                    "margin": margin(control_logits, int(gold[idx])),
                    "mean_js": float(control_metrics["js"].mean()),
                    "js": control_metrics["js"].astype(np.float16),
                    "donor_index": donor_idx if name == "shuffled" else None,
                }
            np.savez_compressed(
                args.save_dir / "blocks" / f"{idx:04d}.npz",
                base=base_attention["q_image"], prior=prior_attention["q_image"],
                base_last=base_attention["last_image"],
                prior_last=prior_attention["last_image"],
                base_mass=base_attention["image_mass"],
                prior_mass=prior_attention["image_mass"],
                **{key: value.astype(np.float16) for key, value in metrics.items()},
                **{f"{name}_js": value["js"] for name, value in comparisons.items()},
            )
            result = {
                "index": idx, "id": str(row["id"]), "question": str(row["question"]),
                "gold": str(row["answer"]), "no_prior_prediction": "ABCD"[base_logits.argmax()],
                "with_prior_prediction": "ABCD"[prior_logits.argmax()],
                "no_prior_correct_logit": float(base_logits[gold[idx]]),
                "with_prior_correct_logit": float(prior_logits[gold[idx]]),
                "no_prior_margin": margin(base_logits, int(gold[idx])),
                "with_prior_margin": margin(prior_logits, int(gold[idx])),
                "delta_margin": margin(prior_logits, int(gold[idx])) - margin(base_logits, int(gold[idx])),
                "no_prior_logits": base_logits.tolist(), "with_prior_logits": prior_logits.tolist(),
                "prior_ids": candidate_indices[idx, top_ranks].tolist(),
                "prior_ranks": top_ranks.tolist(), "prior_utility": selected_utility.tolist(),
                "prior_score": scores[idx, top_ranks].tolist(),
                "image_positions_base": [ordinary["image"][0], ordinary["image"][-1]],
                "image_positions_prior": [shifted["image"][0], shifted["image"][-1]],
                "question_positions_base": ordinary["question"],
                "question_positions_prior": shifted["question"],
                "option_positions_base": ordinary["options"],
                "option_positions_prior": shifted["options"],
                "answer_relevant_last_base": ordinary["last"],
                "answer_relevant_last_prior": shifted["last"],
                "image_grid_thw": ordinary["grid"],
                "visual_attention_mass_base": float(base_attention["image_mass"].mean()),
                "visual_attention_mass_prior": float(prior_attention["image_mass"].mean()),
                "mean_js": float(metrics["js"].mean()),
                "controls": {
                    name: {key: value for key, value in record.items() if key != "js"}
                    for name, record in comparisons.items()
                },
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if (n + 1) % 25 == 0 or n == 0 or n + 1 == count:
                print(f"probed {n + 1}/{count} elapsed={time.monotonic()-started:.1f}s", flush=True)
    if module1_hashes() != before:
        raise RuntimeError("Module 1 source files changed during the probe")
    write_json(args.save_dir / "parity.json", parity)
    reader.close()


if __name__ == "__main__":
    main()
