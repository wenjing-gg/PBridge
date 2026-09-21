"""Summarize paired head statistics, controls and representative visualizations."""

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

from module2_probe.metrics import correlation, normalize, spatial_metrics
from module2_probe.probe_lingshu import load_image, load_rows


def read_cases(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fdr(p_values):
    flat = np.asarray(p_values).reshape(-1)
    sorted_indices = np.argsort(flat)
    out = np.empty_like(flat)
    candidate = flat[sorted_indices] * len(flat) / np.arange(1, len(flat) + 1)
    out[sorted_indices] = np.minimum.accumulate(candidate[::-1])[::-1].clip(0, 1)
    return out.reshape(p_values.shape)


def heatmap(matrix, path, title, cmap, low=None, high=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=low, vmax=high)
    ax.set(xlabel="Attention head", ylabel="LLM layer (0-based)", title=title)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def case_group(record):
    a = record["gold"] == record["no_prior_prediction"]
    b = record["gold"] == record["with_prior_prediction"]
    return "both_correct" if a and b else "fixed" if b else "harmed" if a else "both_wrong"


def case_plot(record, head, block_path, row, path):
    with np.load(block_path) as values:
        base = values["base"].astype(np.float32)[head].mean(axis=0)
        prior = values["prior"].astype(np.float32)[head].mean(axis=0)
    base, prior = normalize(base), normalize(prior)
    grid = record["image_grid_thw"]
    h, w = grid[1] // 2, grid[2] // 2
    if h * w != len(base):
        return
    image = load_image(row["image"])
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].imshow(image)
        axes[1].imshow(image)
        axes[2].imshow(image)
        for ax, array, title, cmap in zip(
            axes, (base, prior, prior - base),
            ("No prior", "PBridge", "PBridge - No prior"),
            ("inferno", "inferno", "coolwarm"),
        ):
            ax.imshow(
                array.reshape(h, w), extent=(0, image.width, image.height, 0),
                cmap=cmap, alpha=0.65, interpolation="nearest",
            )
            ax.set(title=title, xticks=[], yticks=[])
        fig.suptitle(f"Case {record['index']}: {case_group(record)}; layer={head[0]}, head={head[1]}")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    finally:
        image.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, default=Path("module2_probe/outputs/attention_probe"))
    args = parser.parse_args()
    out = args.save_dir
    cases = read_cases(out / "cases.jsonl")
    if not cases:
        raise RuntimeError("No probe cases found")
    array_keys = [
        "js", "l1", "delta_entropy", "entropy_base", "entropy_prior",
        "top5_overlap", "top10_overlap", "top20_overlap",
        "base_mass", "prior_mass", "random_js", "shuffled_js", "zero_js",
    ]
    values = {key: [] for key in array_keys}
    last_js = []
    max_js = []
    for record in cases:
        with np.load(out / "blocks" / f"{record['index']:04d}.npz") as block:
            for key in array_keys:
                values[key].append(block[key].astype(np.float32))
            base = block["base"]
            prior = block["prior"]
            last_js.append(spatial_metrics(base[..., -1:, :], prior[..., -1:, :])["js"])
            max_js.append(spatial_metrics(
                base.max(axis=-2, keepdims=True),
                prior.max(axis=-2, keepdims=True),
            )["js"])
    values = {key: np.stack(data) for key, data in values.items()}
    last_js = np.stack(last_js)
    max_js = np.stack(max_js)
    groups = np.asarray([case_group(case) for case in cases])
    margins = np.asarray([case["delta_margin"] for case in cases])
    utility = np.asarray([case["prior_utility"] for case in cases])
    shape = values["js"].shape[1:]
    correlation_js = np.zeros(shape)
    correlation_l1 = np.zeros(shape)
    p_js = np.ones(shape)
    corr_util_max = np.zeros(shape)
    corr_util_mean = np.zeros(shape)
    for layer in range(shape[0]):
        for head in range(shape[1]):
            point = (slice(None), layer, head)
            correlation_js[layer, head], p_js[layer, head] = correlation(values["js"][point], margins)
            correlation_l1[layer, head], _ = correlation(values["l1"][point], margins)
            corr_util_max[layer, head], _ = correlation(values["js"][point], utility.max(axis=1))
            corr_util_mean[layer, head], _ = correlation(values["js"][point], utility.mean(axis=1))
    adjusted = fdr(p_js)
    with (out / "head_summary.csv").open("w", newline="") as stream:
        fields = [
            "layer", "head", "mean_js", "mean_l1", "mean_entropy_change",
            "corr_js_delta_margin", "p_js", "q_js_bh", "corr_l1_delta_margin",
            "corr_js_utility_max", "corr_js_utility_mean",
            "fix_case_mean_js", "harm_case_mean_js", "both_correct_mean_js",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layer in range(shape[0]):
            for head in range(shape[1]):
                idx = (slice(None), layer, head)
                writer.writerow({
                    "layer": layer, "head": head,
                    "mean_js": float(values["js"][idx].mean()),
                    "mean_l1": float(values["l1"][idx].mean()),
                    "mean_entropy_change": float(values["delta_entropy"][idx].mean()),
                    "corr_js_delta_margin": correlation_js[layer, head],
                    "p_js": p_js[layer, head], "q_js_bh": adjusted[layer, head],
                    "corr_l1_delta_margin": correlation_l1[layer, head],
                    "corr_js_utility_max": corr_util_max[layer, head],
                    "corr_js_utility_mean": corr_util_mean[layer, head],
                    "fix_case_mean_js": float(values["js"][idx][groups == "fixed"].mean())
                    if (groups == "fixed").any() else "",
                    "harm_case_mean_js": float(values["js"][idx][groups == "harmed"].mean())
                    if (groups == "harmed").any() else "",
                    "both_correct_mean_js": float(values["js"][idx][groups == "both_correct"].mean())
                    if (groups == "both_correct").any() else "",
                })
    heatmap(values["js"].mean(axis=0), out / "mean_js.png", "Mean spatial JS", "viridis")
    heatmap(correlation_js, out / "corr_js_margin.png", "Spearman(JS, delta margin)", "coolwarm", -0.35, 0.35)
    if (groups == "fixed").any():
        heatmap(values["js"][groups == "fixed"].mean(axis=0), out / "fixed_js.png",
                "Wrong to correct: spatial JS", "viridis")
    best = np.unravel_index(np.argmax(correlation_js), shape)
    rows = load_rows(json.loads((out / "architecture.json").read_text())["split"])
    figures = out / "case_figures"
    figures.mkdir(exist_ok=True)
    for group in ("fixed", "both_correct", "harmed"):
        selected = np.where(groups == group)[0]
        selected = sorted(selected, key=lambda x: values["js"][x, best[0], best[1]], reverse=True)
        for i in selected[:30 if group == "fixed" else 3]:
            record = cases[i]
            case_plot(record, best, out / "blocks" / f"{record['index']:04d}.npz",
                      rows[record["index"]], figures / f"{group}_{record['index']:04d}.png")
    control = {}
    true_mean = values["js"].mean(axis=(1, 2))
    for name in ("random", "shuffled", "zero"):
        other = values[f"{name}_js"].mean(axis=(1, 2))
        delta = true_mean - other
        control[name] = {
            "mean_js": float(other.mean()),
            "mean_paired_js_difference": float(delta.mean()),
            "p_wilcoxon_two_sided": float(wilcoxon(delta).pvalue),
            "mean_correct_margin": float(np.mean([case["controls"][name]["margin"] for case in cases])),
            "accuracy": float(np.mean([
                np.argmax(case["controls"][name]["logits"]) == "ABCD".index(case["gold"])
                for case in cases
            ])),
        }
    utility_association = {}
    for name, u in [
        ("u_max", utility.max(axis=1)), ("u_mean", utility.mean(axis=1)),
        *[(f"u_top{i + 1}", utility[:, i]) for i in range(3)],
    ]:
        rho, p = correlation(u, true_mean)
        utility_association[name] = {"spearman_mean_js": rho, "p": p}
    utility_association["u_max"]["spearman_delta_margin"], utility_association["u_max"]["margin_p"] = correlation(utility.max(axis=1), margins)
    threshold = np.quantile(utility.max(axis=1), 0.5)
    high = utility.max(axis=1) >= threshold
    utility_association["high_vs_low"] = {
        "threshold": float(threshold),
        "high_n": int(high.sum()),
        "high_mean_js": float(true_mean[high].mean()),
        "low_mean_js": float(true_mean[~high].mean()) if (~high).any() else None,
        "high_mean_delta_margin": float(margins[high].mean()),
        "low_mean_delta_margin": float(margins[~high].mean()) if (~high).any() else None,
    }
    top10 = []
    for idx in np.argsort(correlation_js.reshape(-1))[::-1][:10]:
        layer, head = np.unravel_index(idx, shape)
        top10.append({
            "layer": int(layer), "head": int(head),
            "rho": float(correlation_js[layer, head]), "q": float(adjusted[layer, head]),
            "mean_js": float(values["js"][:, layer, head].mean()),
        })
    mean_head_js = values["js"].mean(axis=0)
    top_shift = [
        {
            "layer": int(np.unravel_index(i, shape)[0]),
            "head": int(np.unravel_index(i, shape)[1]),
            "mean_js": float(mean_head_js.flat[i]),
        }
        for i in np.argsort(mean_head_js.ravel())[::-1][:10]
    ]
    fixed_head_js = values["js"][groups == "fixed"].mean(axis=0) if (
        groups == "fixed"
    ).any() else np.zeros(shape)
    top_fixed = [
        {
            "layer": int(np.unravel_index(i, shape)[0]),
            "head": int(np.unravel_index(i, shape)[1]),
            "fixed_mean_js": float(fixed_head_js.flat[i]),
        }
        for i in np.argsort(fixed_head_js.ravel())[::-1][:10]
    ]
    aggregation = {}
    for name, matrix in (("mean", values["js"]), ("last", last_js), ("max", max_js)):
        global_mean = matrix.mean(axis=(1, 2))
        rho, p = correlation(global_mean, margins)
        aggregation[name] = {
            "mean_js": float(global_mean.mean()),
            "rho_case_mean_js_vs_delta_margin": rho, "p": p,
        }
    summary = {
        "n": len(cases), "group_counts": {
            label: int((groups == label).sum())
            for label in ("fixed", "both_correct", "harmed", "both_wrong")
        },
        "base_accuracy": float(np.mean([
            c["gold"] == c["no_prior_prediction"] for c in cases
        ])),
        "pbridge_accuracy": float(np.mean([
            c["gold"] == c["with_prior_prediction"] for c in cases
        ])),
        "mean_delta_margin": float(margins.mean()),
        "median_delta_margin": float(np.median(margins)),
        "mean_js": float(true_mean.mean()),
        "mean_l1": float(values["l1"].mean()),
        "mean_delta_entropy": float(values["delta_entropy"].mean()),
        "aggregation": aggregation,
        "top10_by_exploratory_corr": top10,
        "top10_by_mean_shift": top_shift,
        "top10_by_fixed_case_shift": top_fixed,
        "mean_visual_attention_mass_base": float(values["base_mass"].mean()),
        "mean_visual_attention_mass_prior": float(values["prior_mass"].mean()),
        "mean_topk_overlap": {
            str(k): float(values[f"top{k}_overlap"].mean()) for k in (5, 10, 20)
        },
        "significant_heads_q05": int((adjusted < 0.05).sum()),
        "max_corr_head": {"layer": int(best[0]), "head": int(best[1]),
                          "rho": float(correlation_js[best]),
                          "p": float(p_js[best]), "q": float(adjusted[best])},
        "mean_js_by_group": {g: float(true_mean[groups == g].mean())
                             if (groups == g).any() else None for g in set(groups)},
        "control": control,
        "utility_association": utility_association,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
