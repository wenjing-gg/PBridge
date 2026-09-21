"""Analyze Image→Prior attention and hidden-state shifts; preselect patch layers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from module2_probe.mechanism_metrics import (
    SHIFT_NAMES,
    bootstrap_mean_ci,
    correlation,
    fdr_bh,
)


CONDITIONS = ("correct", "random", "shuffled")
REGIONS = ("image", "question", "options")


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def heatmap(values, path, title, cmap="viridis", low=None, high=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(
        values, aspect="auto", cmap=cmap, vmin=low, vmax=high
    )
    ax.set(
        xlabel="Attention head",
        ylabel="LLM layer (0-based)",
        title=title,
    )
    fig.colorbar(image, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def group(record):
    gold = record["gold"]
    base = record["no_prior_prediction"]
    correct = record["conditions"]["correct"]["prediction"]
    if base != gold and correct == gold:
        return "fixed"
    if base == gold and correct != gold:
        return "harmed"
    return "both_correct" if base == gold else "both_wrong"


def summarize_attention(root, rng):
    case_records = records(root / "image_to_prior" / "cases.jsonl")
    mass = {condition: [] for condition in CONDITIONS}
    prefix_mean = {condition: [] for condition in CONDITIONS}
    dispersion = {condition: [] for condition in CONDITIONS}
    preference_entropy = {condition: [] for condition in CONDITIONS}
    for record in case_records:
        with np.load(
            root / "image_to_prior" / "blocks"
            / f"{record['index']:04d}.npz"
        ) as archive:
            for condition in CONDITIONS:
                block = archive[condition].astype(np.float32)
                total = block.sum(axis=-1)
                mass[condition].append(total.mean(axis=-1))
                prefix_mean[condition].append(block.mean(axis=-2))
                dispersion[condition].append(total.std(axis=-1))
                preference = block / np.maximum(
                    block.sum(axis=-1, keepdims=True), 1e-12
                )
                entropy = -(
                    preference
                    * np.log(np.clip(preference, 1e-12, None))
                ).sum(axis=-1)
                preference_entropy[condition].append(entropy.mean(axis=-1))
    mass = {key: np.stack(value) for key, value in mass.items()}
    prefix_mean = {
        key: np.stack(value) for key, value in prefix_mean.items()
    }
    dispersion = {
        key: np.stack(value) for key, value in dispersion.items()
    }
    preference_entropy = {
        key: np.stack(value)
        for key, value in preference_entropy.items()
    }
    margins = np.asarray(
        [
            item["conditions"]["correct"]["margin"]
            - item["no_prior_margin"]
            for item in case_records
        ]
    )
    comparisons = {
        "correct_minus_random": mass["correct"] - mass["random"],
        "correct_minus_shuffled": mass["correct"] - mass["shuffled"],
    }
    correlations = {}
    p_values = {}
    sources = {
        "correct_mass": mass["correct"],
        **comparisons,
    }
    for name, values in sources.items():
        rho = np.zeros((28, 28))
        p = np.ones((28, 28))
        for layer in range(28):
            for head in range(28):
                rho[layer, head], p[layer, head] = correlation(
                    values[:, layer, head], margins
                )
        correlations[name] = rho
        p_values[name] = p
    q_values = {
        name: fdr_bh(value) for name, value in p_values.items()
    }
    paired = {}
    for control in ("random", "shuffled"):
        difference = mass["correct"] - mass[control]
        p = np.ones((28, 28))
        for layer in range(28):
            for head in range(28):
                if np.all(difference[:, layer, head] == 0):
                    continue
                p[layer, head] = wilcoxon(
                    difference[:, layer, head]
                ).pvalue
        paired[control] = {
            "mean_difference": difference.mean(axis=0),
            "q": fdr_bh(p),
        }
    summary_dir = root / "analysis"
    summary_dir.mkdir(exist_ok=True)
    heatmap(
        mass["correct"].mean(axis=0),
        summary_dir / "image_prior_correct_mass.png",
        "Correct prior: image→prior attention mass",
    )
    heatmap(
        comparisons["correct_minus_random"].mean(axis=0),
        summary_dir / "image_prior_correct_minus_random.png",
        "Image→prior mass: correct - random",
        "coolwarm",
    )
    heatmap(
        correlations["correct_minus_random"],
        summary_dir / "image_prior_diff_corr_margin.png",
        "Spearman(correct-random image→prior mass, delta margin)",
        "coolwarm",
        -0.3,
        0.3,
    )
    with (summary_dir / "image_prior_head_summary.csv").open(
        "w", newline=""
    ) as stream:
        fields = [
            "layer", "head", "correct_mass", "random_mass",
            "shuffled_mass", "correct_minus_random",
            "correct_minus_shuffled", "corr_correct_margin",
            "q_correct_margin", "corr_correct_random_margin",
            "q_correct_random_margin", "corr_correct_shuffled_margin",
            "q_correct_shuffled_margin", "q_paired_correct_random",
            "q_paired_correct_shuffled", "correct_dispersion",
            "correct_preference_entropy", "prefix1", "prefix2", "prefix3",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layer in range(28):
            for head in range(28):
                writer.writerow(
                    {
                        "layer": layer,
                        "head": head,
                        "correct_mass": mass["correct"][
                            :, layer, head
                        ].mean(),
                        "random_mass": mass["random"][
                            :, layer, head
                        ].mean(),
                        "shuffled_mass": mass["shuffled"][
                            :, layer, head
                        ].mean(),
                        "correct_minus_random": comparisons[
                            "correct_minus_random"
                        ][:, layer, head].mean(),
                        "correct_minus_shuffled": comparisons[
                            "correct_minus_shuffled"
                        ][:, layer, head].mean(),
                        "corr_correct_margin": correlations[
                            "correct_mass"
                        ][layer, head],
                        "q_correct_margin": q_values[
                            "correct_mass"
                        ][layer, head],
                        "corr_correct_random_margin": correlations[
                            "correct_minus_random"
                        ][layer, head],
                        "q_correct_random_margin": q_values[
                            "correct_minus_random"
                        ][layer, head],
                        "corr_correct_shuffled_margin": correlations[
                            "correct_minus_shuffled"
                        ][layer, head],
                        "q_correct_shuffled_margin": q_values[
                            "correct_minus_shuffled"
                        ][layer, head],
                        "q_paired_correct_random": paired["random"][
                            "q"
                        ][layer, head],
                        "q_paired_correct_shuffled": paired["shuffled"][
                            "q"
                        ][layer, head],
                        "correct_dispersion": dispersion["correct"][
                            :, layer, head
                        ].mean(),
                        "correct_preference_entropy": preference_entropy[
                            "correct"
                        ][:, layer, head].mean(),
                        **{
                            f"prefix{prefix + 1}": prefix_mean[
                                "correct"
                            ][:, layer, head, prefix].mean()
                            for prefix in range(3)
                        },
                    }
                )
    layer_rows = []
    for layer in range(28):
        for condition in CONDITIONS:
            values = mass[condition][:, layer].mean(axis=1)
            layer_rows.append(
                {
                    "layer": layer,
                    "condition": condition,
                    "mean_mass": float(values.mean()),
                    "median_mass": float(np.median(values)),
                    "sem_mass": float(
                        values.std(ddof=1) / np.sqrt(len(values))
                    ),
                    "bootstrap95": bootstrap_mean_ci(values, rng),
                }
            )
    result = {
        "n": len(case_records),
        "condition_global_mass": {
            condition: float(mass[condition].mean())
            for condition in CONDITIONS
        },
        "condition_global_preference_entropy": {
            condition: float(preference_entropy[condition].mean())
            for condition in CONDITIONS
        },
        "layer_summary": layer_rows,
        "significant_heads_bh05": {
            name: int((q < 0.05).sum())
            for name, q in q_values.items()
        },
        "paired_specificity_heads_bh05": {
            control: int((value["q"] < 0.05).sum())
            for control, value in paired.items()
        },
        "top_heads_by_correct_random_difference": [],
        "top_heads_by_difference_margin_corr": [],
    }
    difference_mean = comparisons[
        "correct_minus_random"
    ].mean(axis=0)
    for flat in np.argsort(difference_mean.ravel())[::-1][:10]:
        layer, head = np.unravel_index(flat, (28, 28))
        result["top_heads_by_correct_random_difference"].append(
            {
                "layer": int(layer),
                "head": int(head),
                "mean_difference": float(difference_mean[layer, head]),
                "q_paired": float(paired["random"]["q"][layer, head]),
            }
        )
    rho = correlations["correct_minus_random"]
    for flat in np.argsort(rho.ravel())[::-1][:10]:
        layer, head = np.unravel_index(flat, (28, 28))
        result["top_heads_by_difference_margin_corr"].append(
            {
                "layer": int(layer),
                "head": int(head),
                "rho": float(rho[layer, head]),
                "q": float(
                    q_values["correct_minus_random"][layer, head]
                ),
            }
        )
    np.savez_compressed(
        summary_dir / "image_prior_summaries.npz",
        **{
            f"{condition}_mass": value.astype(np.float32)
            for condition, value in mass.items()
        },
        **{
            f"{name}_corr": value.astype(np.float32)
            for name, value in correlations.items()
        },
    )
    (summary_dir / "image_prior_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result


def summarize_hidden(root, rng):
    case_records = records(root / "hidden_shift" / "cases.jsonl")
    shifts = []
    for record in case_records:
        with np.load(
            root / "hidden_shift" / "cases"
            / f"{record['index']:04d}.npz"
        ) as archive:
            shifts.append(archive["shift"].astype(np.float32))
    shifts = np.stack(shifts)
    margins = np.asarray(
        [
            item["conditions"]["correct"]["margin"]
            - item["no_prior_margin"]
            for item in case_records
        ]
    )
    utilities = np.asarray(
        [item["prior_utility"] for item in case_records]
    )
    groups = np.asarray([group(item) for item in case_records])
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    primary = shifts[:, 0, 0, :, 4]
    corr = np.zeros(28)
    p = np.ones(28)
    for layer in range(28):
        corr[layer], p[layer] = correlation(primary[:, layer], margins)
    q = fdr_bh(p)
    discovery = slice(0, 300)
    validation = slice(300, None)
    discovery_corr = np.zeros(28)
    discovery_p = np.ones(28)
    validation_corr = np.zeros(28)
    validation_p = np.ones(28)
    for layer in range(28):
        discovery_corr[layer], discovery_p[layer] = correlation(
            primary[discovery, layer], margins[discovery]
        )
        validation_corr[layer], validation_p[layer] = correlation(
            primary[validation, layer], margins[validation]
        )
    correct_random_discovery = (
        shifts[:300, 0, 0, :, 4]
        - shifts[:300, 1, 0, :, 4]
    ).mean(axis=0)
    correct_shuffled_discovery = (
        shifts[:300, 0, 0, :, 4]
        - shifts[:300, 2, 0, :, 4]
    ).mean(axis=0)
    selected = []
    for layer in np.argsort(discovery_corr)[::-1][:3]:
        selected.append(int(layer))
    selected.append(int(np.argmax(correct_random_discovery)))
    selected.append(int(np.argmax(correct_shuffled_discovery)))
    selected = list(dict.fromkeys(selected))[:5]
    with (analysis / "hidden_layer_summary.csv").open(
        "w", newline=""
    ) as stream:
        fields = [
            "condition", "region", "layer", "metric",
            "mean", "median", "sem", "bootstrap95",
            "corr_delta_margin", "p", "q_bh",
            "fixed_mean", "harmed_mean", "corr_u_max",
            "corr_u_mean", "corr_u_gap",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition_index, condition in enumerate(CONDITIONS):
            for region_index, region in enumerate(REGIONS):
                for metric_index, metric in enumerate(SHIFT_NAMES):
                    p_layer = np.ones(28)
                    rho_layer = np.zeros(28)
                    for layer in range(28):
                        rho_layer[layer], p_layer[layer] = correlation(
                            shifts[
                                :, condition_index, region_index,
                                layer, metric_index
                            ],
                            margins,
                        )
                    q_layer = fdr_bh(p_layer)
                    for layer in range(28):
                        values = shifts[
                            :, condition_index, region_index,
                            layer, metric_index
                        ]
                        utility_correlations = [
                            correlation(values, utilities.max(axis=1))[0],
                            correlation(values, utilities.mean(axis=1))[0],
                            correlation(
                                values,
                                utilities[:, 0] - utilities[:, 1],
                            )[0],
                        ]
                        writer.writerow(
                            {
                                "condition": condition,
                                "region": region,
                                "layer": layer,
                                "metric": metric,
                                "mean": values.mean(),
                                "median": np.median(values),
                                "sem": values.std(ddof=1)
                                / np.sqrt(len(values)),
                                "bootstrap95": json.dumps(
                                    bootstrap_mean_ci(values, rng)
                                    if (
                                        condition == "correct"
                                        and region == "image"
                                        and metric == "token_cosine"
                                    )
                                    else None
                                ),
                                "corr_delta_margin": rho_layer[layer],
                                "p": p_layer[layer],
                                "q_bh": q_layer[layer],
                                "fixed_mean": values[
                                    groups == "fixed"
                                ].mean(),
                                "harmed_mean": values[
                                    groups == "harmed"
                                ].mean(),
                                "corr_u_max": utility_correlations[0],
                                "corr_u_mean": utility_correlations[1],
                                "corr_u_gap": utility_correlations[2],
                            }
                        )
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition_index, condition in enumerate(CONDITIONS):
        values = shifts[:, condition_index, 0, :, 4]
        ax.plot(
            range(28), values.mean(axis=0),
            label=condition.capitalize(),
        )
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        ax.fill_between(
            range(28),
            values.mean(axis=0) - sem,
            values.mean(axis=0) + sem,
            alpha=0.2,
        )
    ax.set(
        xlabel="LLM layer (0-based)",
        ylabel="Mean token-wise cosine shift",
        title="Visual hidden-state shift",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis / "visual_hidden_shift.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(28), corr, marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="LLM layer (0-based)",
        ylabel="Spearman rho",
        title="Visual shift vs correct-prior margin gain",
    )
    fig.tight_layout()
    fig.savefig(analysis / "visual_shift_corr_margin.png", dpi=150)
    plt.close(fig)
    primary_utility = {}
    for name, value in {
        "u_max": utilities.max(axis=1),
        "u_mean": utilities.mean(axis=1),
        "u_top1": utilities[:, 0],
        "u_gap": utilities[:, 0] - utilities[:, 1],
    }.items():
        correlations = [
            correlation(primary[:, layer], value)[0]
            for layer in range(28)
        ]
        primary_utility[name] = {
            "max_abs_rho": float(
                correlations[int(np.argmax(np.abs(correlations)))]
            ),
            "layer": int(np.argmax(np.abs(correlations))),
        }
    result = {
        "n": len(case_records),
        "group_counts": {
            label: int((groups == label).sum())
            for label in (
                "fixed", "both_correct", "harmed", "both_wrong"
            )
        },
        "primary_metric": "image token-wise cosine shift",
        "selected_layers_from_first_300": selected,
        "selected_layer_validation": [
            {
                "layer": layer,
                "discovery_rho": float(discovery_corr[layer]),
                "discovery_p": float(discovery_p[layer]),
                "validation_rho": float(validation_corr[layer]),
                "validation_p": float(validation_p[layer]),
                "full_rho": float(corr[layer]),
                "full_q": float(q[layer]),
                "mean_correct": float(primary[:, layer].mean()),
                "mean_random": float(
                    shifts[:, 1, 0, layer, 4].mean()
                ),
                "mean_shuffled": float(
                    shifts[:, 2, 0, layer, 4].mean()
                ),
                "fixed_mean": float(
                    primary[groups == "fixed", layer].mean()
                ),
                "harmed_mean": float(
                    primary[groups == "harmed", layer].mean()
                ),
            }
            for layer in selected
        ],
        "global_primary_shift": {
            condition: float(
                shifts[:, index, 0, :, 4].mean()
            )
            for index, condition in enumerate(CONDITIONS)
        },
        "utility_relationship": primary_utility,
        "cka": "not run: exact token-level linear CKA would dominate compute/storage; cosine and normalized-L2 were computed for mean, max, and token-wise pooling",
    }
    (analysis / "hidden_shift_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (analysis / "candidate_layers.json").write_text(
        json.dumps(
            {
                "discovery_cases": [0, 299],
                "validation_cases": [300, 412],
                "selection_rule": (
                    "top-3 discovery Spearman layers plus maximum "
                    "correct-random and correct-shuffled mean excess"
                ),
                "layers": selected,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        analysis / "hidden_shift_arrays.npz",
        shift=shifts,
        delta_margin=margins,
        groups=groups,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("module2_probe/outputs/mechanism"),
    )
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    attention = summarize_attention(args.root, rng)
    hidden = summarize_hidden(args.root, rng)
    combined = {
        "image_to_prior": attention,
        "hidden_shift": hidden,
    }
    (args.root / "analysis" / "summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "image_to_prior_global_mass": attention[
                    "condition_global_mass"
                ],
                "attention_specific_heads": attention[
                    "paired_specificity_heads_bh05"
                ],
                "candidate_layers": hidden[
                    "selected_layers_from_first_300"
                ],
                "validation": hidden[
                    "selected_layer_validation"
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
