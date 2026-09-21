"""Quantify recoverable Top-3 harm before learning a consistency scorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from module2_alignment.common import OUT, margin, write_json


def main():
    root = OUT / "counterfactual"
    chunks = list(root.glob("top3_shard*.npz"))
    if not chunks:
        raise RuntimeError("No counterfactual chunks found")
    fields = ["index", "gold", "no_prior_logits", "top3_logits", "ranks", "utility", "ppr_score"]
    arrays = {field: [] for field in fields}
    for path in chunks:
        with np.load(path) as chunk:
            for field in fields:
                arrays[field].append(chunk[field])
    arrays = {field: np.concatenate(parts) for field, parts in arrays.items()}
    order = np.argsort(arrays["index"])
    arrays = {field: value[order] for field, value in arrays.items()}
    if not np.array_equal(arrays["index"], np.arange(1654)):
        raise RuntimeError("Need all 1654 unique original training cases")
    split = json.loads((root / "split.json").read_text())
    if set(split["train_indices"]) & set(split["dev_indices"]):
        raise RuntimeError("Module 2 train and dev overlap")
    gold = arrays["gold"]
    base = arrays["no_prior_logits"]
    top3 = arrays["top3_logits"]
    baseline_margin = margin(base, gold)
    prior_margin = margin(top3, gold)
    delta = prior_margin - baseline_margin
    base_pred = base.argmax(axis=1)
    prior_pred = top3.argmax(axis=1)
    # An oracle based on answer correctness is a separate, stronger bound.
    oracle_margin_pred = np.where((delta > 0), prior_pred, base_pred)
    oracle_acc_pred = np.where(
        (prior_pred == gold) & (base_pred != gold), prior_pred, base_pred
    )
    result = {}
    for name, indices in (("module2_train", split["train_indices"]), ("module2_dev", split["dev_indices"])):
        idx = np.asarray(indices, dtype=np.int64)
        baseline_ok = base_pred[idx] == gold[idx]
        top3_ok = prior_pred[idx] == gold[idx]
        oracle_ok = oracle_margin_pred[idx] == gold[idx]
        fix = int((~baseline_ok & top3_ok).sum())
        harm = int((baseline_ok & ~top3_ok).sum())
        positive = int((delta[idx] > 0).sum())
        negative = int((delta[idx] < 0).sum())
        result[name] = {
            "n": len(idx),
            "base_correct": int(baseline_ok.sum()),
            "top3_correct": int(top3_ok.sum()),
            "top3_fix": fix,
            "top3_harm": harm,
            "top3_net": fix - harm,
            "delta_margin_mean": float(delta[idx].mean()),
            "delta_margin_median": float(np.median(delta[idx])),
            "margin_helpful": positive,
            "margin_harmful": negative,
            "margin_neutral": len(idx) - positive - negative,
            "margin_oracle_correct": int(oracle_ok.sum()),
            "margin_oracle_fix": int((~baseline_ok & oracle_ok).sum()),
            "margin_oracle_harm": int((baseline_ok & ~oracle_ok).sum()),
            "accuracy_oracle_correct": int((oracle_acc_pred[idx] == gold[idx]).sum()),
            "top3_vs_base_p_two_sided": float(
                binomtest(fix, fix + harm).pvalue
            ) if fix + harm else 1.0,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "top3_merged.npz",
        **arrays,
        base_margin=baseline_margin,
        top3_margin=prior_margin,
        delta_margin=delta,
    )
    write_json(root / "oracle_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
