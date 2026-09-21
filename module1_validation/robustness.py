"""Robustness runs for the default predicted-score-weighted PBridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module1_ppr"))

from model import PPRMoE, TOP_CONTEXT
from reader import FrozenReader
from run import convergence, load_assets, load_rows, stage1


OUT = ROOT / "module1_validation" / "robustness_outputs"
CONFIGS = {
    "full_seed19": {"model_seed": 20260919, "split_seed": None},
    "full_seed20": {"model_seed": 20260920, "split_seed": None},
    "group_seed20": {"model_seed": 20260918, "split_seed": 20260920},
    "group_seed21": {"model_seed": 20260918, "split_seed": 20260921},
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(rows: list[dict], split_seed: int | None) -> dict:
    if split_seed is None:
        indices = np.arange(len(rows), dtype=np.int64)
        return {
            "kind": "full",
            "train_indices": indices.tolist(),
            "heldout_indices": [],
            "train_images": len({str(row["img_id"]) for row in rows}),
            "heldout_images": 0,
            "image_overlap": 0,
        }
    groups = np.asarray([str(row["img_id"]) for row in rows])
    unique = np.unique(groups)
    rng = np.random.default_rng(split_seed)
    heldout_groups = set(
        rng.permutation(unique)[: round(0.2 * len(unique))].tolist()
    )
    heldout = np.asarray(
        [i for i, group in enumerate(groups) if group in heldout_groups],
        dtype=np.int64,
    )
    train = np.asarray(
        [i for i, group in enumerate(groups) if group not in heldout_groups],
        dtype=np.int64,
    )
    train_groups = set(groups[train].tolist())
    heldout_group_values = set(groups[heldout].tolist())
    return {
        "kind": "image_group_80_20",
        "split_seed": split_seed,
        "train_indices": train.tolist(),
        "heldout_indices": heldout.tolist(),
        "train_images": len(train_groups),
        "heldout_images": len(heldout_group_values),
        "image_overlap": len(train_groups & heldout_group_values),
    }


def prepare(config_name: str, device_name: str) -> None:
    config = CONFIGS[config_name]
    set_seed(config["model_seed"])
    rows = load_rows("train")
    split = split_indices(rows, config["split_seed"])
    indices = np.asarray(split["train_indices"], dtype=np.int64)
    assets = load_assets("train")
    subset = {key: value[indices] for key, value in assets.items()}
    model = PPRMoE(3584).to(device_name)
    result = stage1(model, subset, torch.device(device_name))
    if not result["converged"]:
        raise RuntimeError(f"{config_name}: Stage 1 did not converge")
    output = OUT / config_name
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stage1": result,
            "model_seed": config["model_seed"],
        },
        output / "stage1.pt",
    )
    write_json(output / "split.json", split)
    write_json(
        output / "stage1_summary.json",
        {
            "model_seed": config["model_seed"],
            "split_seed": config["split_seed"],
            "train_rows": len(indices),
            "heldout_rows": len(split["heldout_indices"]),
            **{
                key: value
                for key, value in result.items()
                if not key.endswith("_history")
            },
        },
    )


def evaluate_split(
    model: PPRMoE,
    reader: FrozenReader,
    batch_size: int,
    split_name: str,
    indices: np.ndarray,
) -> np.ndarray:
    assets = load_assets(split_name)
    rows = load_rows(split_name)
    device = reader.model_device
    query = torch.from_numpy(assets["query_embeddings"][indices]).to(device)
    knowledge = torch.from_numpy(
        assets["candidate_embeddings"][indices]
    ).to(device)
    proxy = torch.from_numpy(assets["utility_proxy"][indices]).to(device)
    model.eval()
    with torch.no_grad():
        scores, _, _ = model(query, knowledge, proxy)
        ranks = scores.topk(TOP_CONTEXT, dim=-1).indices
        prefixes = model.prefix_embeddings(
            knowledge,
            scores,
            ranks,
        )
    logits = np.empty((len(indices), 4), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        part = indices[start:end]
        with torch.no_grad():
            logits[start:end] = reader.logits(
                rows,
                [(int(index), []) for index in part],
                prefixes=prefixes[start:end],
            ).float().cpu().numpy()
    return logits


def train_stage2(
    config_name: str,
    device_name: str,
    batch_size: int,
) -> None:
    output = OUT / config_name
    split = json.loads((output / "split.json").read_text())
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    initial = torch.load(
        output / "stage1.pt", map_location="cpu", weights_only=True
    )
    seed = int(initial["model_seed"])
    set_seed(seed)
    reader = FrozenReader(device_name)
    device = reader.model_device
    rows = load_rows("train")
    assets = load_assets("train")
    data = {
        key: torch.from_numpy(assets[key]).to(device)
        for key in (
            "query_embeddings",
            "candidate_embeddings",
            "utility_proxy",
            "utility_target",
            "gold",
        )
    }
    model = PPRMoE(reader.hidden_size).to(device)
    model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    best_loss = float("inf")
    best_state = None
    history = []
    started = time.monotonic()
    for epoch in range(1, 101):
        model.train()
        epoch_values = [[], [], [], []]
        for start in range(0, len(train_indices), batch_size):
            part = train_indices[start : start + batch_size]
            idx = torch.as_tensor(part, dtype=torch.long, device=device)
            scores, routing, chosen = model(
                data["query_embeddings"][idx],
                data["candidate_embeddings"][idx],
                data["utility_proxy"][idx],
            )
            ranks = scores.detach().topk(TOP_CONTEXT, dim=-1).indices
            prefixes = model.prefix_embeddings(
                data["candidate_embeddings"][idx],
                scores,
                ranks,
            )
            logits = reader.logits(
                rows,
                [(int(index), []) for index in part],
                prefixes=prefixes,
                require_grad=True,
            )
            ce = F.cross_entropy(logits.float(), data["gold"][idx])
            rank = model.rank_loss(scores, data["utility_target"][idx])
            balance = model.balance_loss(routing, chosen)
            total = ce + rank + 0.01 * balance
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for values, value in zip(
                epoch_values, (total, ce, rank, balance), strict=True
            ):
                values.append(float(value.detach().cpu()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_values[0])),
            "ce": float(np.mean(epoch_values[1])),
            "rank": float(np.mean(epoch_values[2])),
            "balance": float(np.mean(epoch_values[3])),
            "elapsed_s": time.monotonic() - started,
        }
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        summary = convergence(
            [item["loss"] for item in history],
            minimum=4,
            window=3,
            relative_tolerance=0.01,
            absolute_tolerance=0.005,
        )
        print(
            f"{config_name} epoch={epoch} "
            f"loss={row['loss']:.6f} ce={row['ce']:.6f}",
            flush=True,
        )
        if summary["converged"]:
            break
    if best_state is None or not summary["converged"]:
        raise RuntimeError(f"{config_name}: Stage 2 did not converge")
    model.load_state_dict(best_state)
    model.to(device)
    test_indices = np.arange(len(load_rows("test")), dtype=np.int64)
    logits = evaluate_split(
        model, reader, batch_size, "test", test_indices
    )
    test_gold = np.asarray(
        ["ABCD".index(str(row["answer"]).strip().upper())
         for row in load_rows("test")],
        dtype=np.int64,
    )
    test_prediction = logits.argmax(axis=1)
    np.save(output / "test_logits.npy", logits)
    heldout_indices = np.asarray(split["heldout_indices"], dtype=np.int64)
    heldout_result = None
    if len(heldout_indices):
        heldout_logits = evaluate_split(
            model,
            reader,
            batch_size,
            "train",
            heldout_indices,
        )
        heldout_gold = assets["gold"][heldout_indices]
        heldout_prediction = heldout_logits.argmax(axis=1)
        np.save(output / "heldout_logits.npy", heldout_logits)
        heldout_result = {
            "correct": int((heldout_prediction == heldout_gold).sum()),
            "accuracy": float((heldout_prediction == heldout_gold).mean()),
        }
    write_json(
        output / "training_summary.json",
        {
            "config": config_name,
            "prefix_weighting": "softmax(PPR predicted scores / 0.5)",
            "model_seed": seed,
            "split_seed": CONFIGS[config_name]["split_seed"],
            "train_rows": len(train_indices),
            "converged": True,
            "epochs": len(history),
            "best_epoch": int(np.argmin([row["loss"] for row in history]) + 1),
            "best_loss": best_loss,
            "test_correct": int((test_prediction == test_gold).sum()),
            "test_accuracy": float((test_prediction == test_gold).mean()),
            "heldout": heldout_result,
            "history": history,
        },
    )
    reader.close()


def analyze() -> None:
    rows = load_rows("test")
    gold = np.asarray(
        ["ABCD".index(str(row["answer"]).strip().upper()) for row in rows],
        dtype=np.int64,
    )
    no_rag = np.load(
        ROOT / "module1_ppr" / "artifacts" / "lingshu_7b"
        / "no_rag_predictions.npz"
    )["probabilities"].argmax(axis=1)
    def compare(
        prediction: np.ndarray,
        baseline: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        predicted_ok = prediction == labels
        baseline_ok = baseline == labels
        fix = int((predicted_ok & ~baseline_ok).sum())
        harm = int((~predicted_ok & baseline_ok).sum())
        return {
            "n": int(len(labels)),
            "no_rag_correct": int(baseline_ok.sum()),
            "no_rag_accuracy": float(baseline_ok.mean()),
            "pbridge_correct": int(predicted_ok.sum()),
            "pbridge_accuracy": float(predicted_ok.mean()),
            "fix": fix,
            "harm": harm,
            "net_questions": fix - harm,
            "net_gain_pp": float(
                100 * (predicted_ok.mean() - baseline_ok.mean())
            ),
            "paired_exact_p": float(binomtest(fix, fix + harm).pvalue),
        }

    official = np.load(
        ROOT / "module1_ppr" / "artifacts" / "lingshu_7b"
        / "predictions.npz"
    )["probabilities"].argmax(axis=1)
    results = [
        {
            "config": "default_seed18",
            "model_seed": 20260918,
            "split_seed": None,
            "split_kind": "full",
            "train_rows": 1654,
            "heldout_rows": 0,
            "image_overlap": 0,
            "test": compare(official, no_rag, gold),
        }
    ]
    for config_name, config in CONFIGS.items():
        predicted = np.load(
            OUT / config_name / "test_logits.npy"
        ).argmax(axis=1)
        split = json.loads((OUT / config_name / "split.json").read_text())
        results.append(
            {
                "config": config_name,
                "model_seed": config["model_seed"],
                "split_seed": config["split_seed"],
                "split_kind": split["kind"],
                "train_rows": len(split["train_indices"]),
                "heldout_rows": len(split["heldout_indices"]),
                "image_overlap": split["image_overlap"],
                "test": compare(predicted, no_rag, gold),
            }
        )
        heldout_indices = np.asarray(
            split["heldout_indices"], dtype=np.int64
        )
        if len(heldout_indices):
            train_assets = load_assets("train")
            heldout_gold = train_assets["gold"][heldout_indices]
            heldout_no_rag = train_assets[
                "no_rag_probabilities"
            ][heldout_indices].argmax(axis=1)
            heldout_predicted = np.load(
                OUT / config_name / "heldout_logits.npy"
            ).argmax(axis=1)
            results[-1]["heldout"] = compare(
                heldout_predicted, heldout_no_rag, heldout_gold
            )
    write_json(
        OUT / "summary.json",
        {
            "no_rag_correct": int((no_rag == gold).sum()),
            "no_rag_accuracy": float((no_rag == gold).mean()),
            "runs": results,
        },
    )
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", choices=CONFIGS, required=True)
    prepare_parser.add_argument("--device", default="cuda:0")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", choices=CONFIGS, required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--batch-size", type=int, default=4)
    subparsers.add_parser("analyze")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    if args.command == "prepare":
        prepare(args.config, args.device)
    elif args.command == "train":
        train_stage2(args.config, args.device, args.batch_size)
    else:
        analyze()


if __name__ == "__main__":
    main()
