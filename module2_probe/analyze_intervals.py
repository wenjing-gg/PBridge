"""Summarize post-hoc multi-layer patching separately from prespecified tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from module2_probe.mechanism_metrics import bootstrap_mean_ci


def read(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def group(item):
    if (
        item["no_prior_prediction"] != item["gold"]
        and item["correct_prediction"] == item["gold"]
    ):
        return "fixed"
    if (
        item["no_prior_prediction"] == item["gold"]
        and item["correct_prediction"] != item["gold"]
    ):
        return "harmed"
    return (
        "both_correct"
        if item["no_prior_prediction"] == item["gold"]
        else "both_wrong"
    )


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
            "interval_patching_summary.json"
        ),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(20260925)
    all_rows = []
    for path in sorted(args.patch_dir.glob("intervals_*.jsonl")):
        rows = read(path)
        groups = np.asarray([group(item) for item in rows])
        for kind in rows[0]["patches"]:
            for interval in rows[0]["patches"][kind]:
                values = np.asarray(
                    [
                        item["patches"][kind][interval][
                            "recovery_loss"
                        ]
                        for item in rows
                    ]
                )
                fixed = values[groups == "fixed"]
                all_rows.append(
                    {
                        "kind": kind,
                        "interval": interval,
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "bootstrap95": bootstrap_mean_ci(values, rng),
                        "p_greater_zero": float(
                            wilcoxon(
                                values, alternative="greater"
                            ).pvalue
                        )
                        if not np.all(values == 0)
                        else 1.0,
                        "fixed_n": int(len(fixed)),
                        "fixed_mean": float(fixed.mean()),
                        "fixed_p_greater_zero": float(
                            wilcoxon(
                                fixed, alternative="greater"
                            ).pvalue
                        )
                        if not np.all(fixed == 0)
                        else 1.0,
                    }
                )
    result = {
        "status": (
            "post-hoc supportive analysis after viewing single-layer "
            "validation effects; not an independent confirmatory test"
        ),
        "rows": all_rows,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
