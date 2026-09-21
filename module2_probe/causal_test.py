"""Head-specific visual-key masking versus matched random-key controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from module2_probe.attention_hooks import Intervention
from module2_probe.metrics import normalize, correlation
from module2_probe.probe_lingshu import (
    FrozenReader, load_ppr, load_rows, margin, run, spans,
)


def read_cases(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def group(case):
    a = case["gold"] == case["no_prior_prediction"]
    b = case["gold"] == case["with_prior_prediction"]
    return "fixed" if b and not a else "harmed" if a and not b else (
        "both_correct" if a else "both_wrong"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("module1_ppr/artifacts/lingshu_7b/ppr_moe.pt"))
    parser.add_argument("--save-dir", type=Path, default=Path("module2_probe/outputs/attention_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--discovery-count", type=int, default=300)
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1, help="1-based rank by discovery JS/margin correlation")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    cases = read_cases(args.save_dir / "cases.jsonl")
    if len(cases) <= args.discovery_count:
        raise ValueError("Need an independent holdout after head discovery")
    discovery = cases[:args.discovery_count]
    validation = cases[args.discovery_count:]
    js_discovery = []
    for case in discovery:
        with np.load(args.save_dir / "blocks" / f"{case['index']:04d}.npz") as block:
            js_discovery.append(block["js"].astype(np.float32))
    js_discovery = np.stack(js_discovery)
    delta = np.array([case["delta_margin"] for case in discovery])
    corr = np.zeros(js_discovery.shape[1:])
    for layer in range(corr.shape[0]):
        for head in range(corr.shape[1]):
            corr[layer, head], _ = correlation(js_discovery[:, layer, head], delta)
    if not 1 <= args.rank <= corr.size:
        raise ValueError("Invalid discovery head rank")
    best = np.unravel_index(np.argsort(corr.ravel())[-args.rank], corr.shape)
    test_js = []
    for case in validation:
        with np.load(args.save_dir / "blocks" / f"{case['index']:04d}.npz") as block:
            test_js.append(float(block["js"][best]))
    test_rho, test_p = correlation(
        np.array(test_js), np.array([case["delta_margin"] for case in validation])
    )
    selected = []
    # Include corrections and matched examples without choosing on the probe's shift.
    for label in ("fixed", "both_correct", "harmed", "both_wrong"):
        members = [case for case in validation if group(case) == label]
        selected.extend(members[: max(1, args.max_cases // 4)])
    selected = selected[:args.max_cases]
    reader = FrozenReader(args.device)
    data, scores, ranks, prefixes = load_ppr(reader, args.checkpoint, "test")
    rows = load_rows("test")
    rng = np.random.default_rng(args.seed)
    results = []
    for case in selected:
        idx = case["index"]
        row = rows[idx]
        positions = spans(reader, row)
        positions["image"] = [x + 3 for x in positions["image"]]
        positions["question"] = [x + 3 for x in positions["question"]]
        positions["last"] += 3
        with np.load(args.save_dir / "blocks" / f"{idx:04d}.npz") as block:
            base = normalize(block["base"][best].astype(np.float32).mean(axis=0))
            prior = normalize(block["prior"][best].astype(np.float32).mean(axis=0))
        k = min(args.top_k, len(base))
        targeted = np.argsort(prior - base)[-k:]
        chosen = [positions["image"][x] for x in targeted]
        pair = prefixes[idx : idx + 1]
        expected_margin = case["with_prior_margin"]
        with torch.no_grad():
            native = reader.logits([row], [(0, [])], prefixes=pair)[0].float().cpu().numpy()
        if abs(margin(native, "ABCD".index(case["gold"])) - expected_margin) > 1e-3:
            raise RuntimeError(f"Prior output mismatch in causal test: {idx}")
        masked, _ = run(
            reader, row, pair, positions,
            Intervention(int(best[0]), int(best[1]), tuple(chosen)), capture=False,
        )
        gold = "ABCD".index(case["gold"])
        targeted_drop = expected_margin - margin(masked, gold)
        random_drops = []
        for _ in range(args.random_repeats):
            sampled = rng.choice(positions["image"], size=k, replace=False).tolist()
            masked_random, _ = run(
                reader, row, pair, positions,
                Intervention(int(best[0]), int(best[1]), tuple(sampled)), capture=False,
            )
            random_drops.append(expected_margin - margin(masked_random, gold))
        results.append({
            "index": idx, "id": case["id"], "group": group(case),
            "layer": int(best[0]), "head": int(best[1]),
            "targeted_image_offsets": targeted.tolist(),
            "targeted_drop": float(targeted_drop),
            "random_drops": [float(x) for x in random_drops],
            "random_mean_drop": float(np.mean(random_drops)),
        })
        print(f"causal {len(results)}/{len(selected)} case={idx} selected_drop={targeted_drop:.4f} random_mean={np.mean(random_drops):.4f}", flush=True)
    suffix = "" if args.rank == 1 else f"_rank{args.rank}"
    with (args.save_dir / f"causal_cases{suffix}.jsonl").open("w") as file:
        for record in results:
            file.write(json.dumps(record) + "\n")
    targeted = np.array([x["targeted_drop"] for x in results])
    random = np.array([x["random_mean_drop"] for x in results])
    paired = targeted - random
    result = {
        "selected_by": f"max positive JS/delta-margin Spearman on first {args.discovery_count} cases",
        "discovery_head_rank": args.rank,
        "discovery_corr": float(corr[best]), "validation_cases": len(validation),
        "validation_corr": test_rho, "validation_corr_p": test_p,
        "layer": int(best[0]), "head": int(best[1]), "k": args.top_k,
        "random_repeats": args.random_repeats,
        "causal_cases": len(results),
        "group_counts": {g: sum(x["group"] == g for x in results)
                         for g in ("fixed", "both_correct", "harmed", "both_wrong")},
        "selected_mean_margin_drop": float(targeted.mean()),
        "random_mean_margin_drop": float(random.mean()),
        "mean_paired_drop_difference": float(paired.mean()),
        "p_wilcoxon_two_sided": float(wilcoxon(paired).pvalue),
        "p_wilcoxon_selected_greater": float(wilcoxon(paired, alternative="greater").pvalue),
        "selected_greater_count": int((paired > 0).sum()),
        "group_mean_drop": {
            g: {
                "selected": float(targeted[[x["group"] == g for x in results]].mean()),
                "random": float(random[[x["group"] == g for x in results]].mean()),
            } for g in ("fixed", "both_correct", "harmed", "both_wrong")
            if any(x["group"] == g for x in results)
        },
    }
    (args.save_dir / f"causal_summary{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    reader.close()


if __name__ == "__main__":
    main()
