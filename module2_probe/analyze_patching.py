"""Summarize held-out activation-patching effects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from module2_probe.mechanism_metrics import bootstrap_mean_ci, fdr_bh


def records(paths):
    merged = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                entry = merged.setdefault(
                    record["index"],
                    {
                        key: value
                        for key, value in record.items()
                        if key != "patches"
                    },
                )
                entry.setdefault("patches", {}).update(record["patches"])
    return [merged[index] for index in sorted(merged)]


def group(record):
    gold = record["gold"]
    base = record["no_prior_prediction"]
    correct = record["correct_prediction"]
    if base != gold and correct == gold:
        return "fixed"
    if base == gold and correct != gold:
        return "harmed"
    return "both_correct" if base == gold else "both_wrong"


def test_greater(values):
    values = np.asarray(values)
    if np.all(values == 0):
        return 1.0
    return float(wilcoxon(values, alternative="greater").pvalue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=Path("module2_probe/outputs/mechanism/patching"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "module2_probe/outputs/mechanism/analysis/"
            "patching_summary.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    paths = sorted(
        path for path in args.patch_dir.glob("*.jsonl")
        if not path.name.startswith("intervals_")
    )
    data = records(paths)
    if len(data) != 113:
        raise RuntimeError(f"Expected 113 patch cases, got {len(data)}")
    rng = np.random.default_rng(args.seed)
    kinds = sorted(data[0]["patches"])
    layers = sorted(
        int(layer)
        for layer in next(iter(data[0]["patches"].values()))
    )
    groups = np.asarray([group(item) for item in data])
    rows = []
    for kind in kinds:
        p_values = []
        layer_values = {}
        for layer in layers:
            values = np.asarray(
                [
                    item["patches"][kind][str(layer)]["recovery_loss"]
                    for item in data
                ]
            )
            layer_values[layer] = values
            p_values.append(test_greater(values))
        q_values = fdr_bh(p_values)
        for position, layer in enumerate(layers):
            values = layer_values[layer]
            row = {
                "kind": kind,
                "layer": layer,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "sem": float(
                    values.std(ddof=1) / np.sqrt(len(values))
                ),
                "bootstrap95": bootstrap_mean_ci(values, rng),
                "p_greater_zero": p_values[position],
                "q_bh": float(q_values[position]),
                "positive_count": int((values > 0).sum()),
                "fixed_mean": float(
                    values[groups == "fixed"].mean()
                ) if (groups == "fixed").any() else None,
                "fixed_p_greater_zero": test_greater(
                    values[groups == "fixed"]
                ) if (groups == "fixed").any() else None,
                "harmed_mean": float(
                    values[groups == "harmed"].mean()
                ) if (groups == "harmed").any() else None,
                "both_correct_mean": float(
                    values[groups == "both_correct"].mean()
                ),
                "both_wrong_mean": float(
                    values[groups == "both_wrong"].mean()
                ),
            }
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    best = {
        kind: max(
            (row for row in rows if row["kind"] == kind),
            key=lambda row: row["mean"],
        )
        for kind in kinds
    }
    pairwise = {}
    for layer in layers:
        if "image" in kinds and "question" in kinds:
            image = np.asarray(
                [
                    item["patches"]["image"][str(layer)][
                        "recovery_loss"
                    ]
                    for item in data
                ]
            )
            question = np.asarray(
                [
                    item["patches"]["question"][str(layer)][
                        "recovery_loss"
                    ]
                    for item in data
                ]
            )
            difference = question - image
            pairwise[str(layer)] = {
                "question_minus_image_mean": float(difference.mean()),
                "p_question_greater": test_greater(difference),
            }
    result = {
        "n": len(data),
        "group_counts": {
            label: int((groups == label).sum())
            for label in (
                "fixed", "both_correct", "harmed", "both_wrong"
            )
        },
        "candidate_layers": layers,
        "delta_margin_mean": float(
            np.mean([item["delta_margin"] for item in data])
        ),
        "best_by_patch_kind": best,
        "question_vs_image": pairwise,
        "all_rows": rows,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "group_counts": result["group_counts"],
                "best_by_patch_kind": best,
                "question_vs_image": pairwise,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
