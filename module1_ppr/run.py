"""Train and evaluate PBridge with the frozen MedMO-8B reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from model import PPRMoE, TOP_CONTEXT
from reader import (
    FrozenReader,
    LABELS,
    MEDMO8B_DIR,
    READER_KEY,
    READER_NAME,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MAX_STAGE1_EPOCHS = 12000
STAGE1_MIN_EPOCHS = 200
STAGE1_WINDOW = 25
STAGE2_MAX_EPOCHS = 100
STAGE2_MIN_EPOCHS = 4
STAGE2_WINDOW = 3


def load_assets(split: str) -> dict[str, np.ndarray]:
    path = ARTIFACTS / READER_KEY / f"{split}_assets.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run reader_assets.py first")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def convergence(
    values: list[float],
    *,
    minimum: int,
    window: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {
        "converged": False,
        "epochs": int(len(array)),
        "final_loss": float(array[-1]),
        "best_loss": float(array.min()),
        "best_epoch": int(array.argmin() + 1),
        "min_epochs": int(minimum),
        "window": int(window),
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
    }
    if len(array) < minimum + 2 * window:
        return result
    previous = array[-2 * window : -window].mean()
    recent = array[-window:].mean()
    absolute = previous - recent
    relative = absolute / max(abs(previous), 1e-12)
    result.update(
        {
            "previous_window_mean": float(previous),
            "recent_window_mean": float(recent),
            "absolute_improvement": float(absolute),
            "relative_improvement": float(relative),
            "converged": bool(
                abs(absolute) <= absolute_tolerance
                and abs(relative) <= relative_tolerance
            ),
        }
    )
    return result


def dataset_path(split: str) -> Path:
    return Path(
        "/nfsdata_a40/cyf/shared_data/pediatric_vqa/processed/"
        f"pediatrics_mqa_module1_8_2/{split}.parquet"
    )


def load_rows(split: str) -> list[dict[str, Any]]:
    return pq.read_table(dataset_path(split)).to_pylist()


def answer_indices(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [LABELS.index(str(row["answer"]).strip().upper()) for row in rows],
        dtype=np.int64,
    )


def evaluate_probabilities(
    probabilities: np.ndarray,
    gold: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    logp = np.log(np.clip(probabilities, 1e-12, 1.0))
    one_hot = np.eye(4, dtype=np.float32)[gold]
    return {
        "n": int(len(gold)),
        "correct": int(np.sum(predictions == gold)),
        "accuracy": float(np.mean(predictions == gold)),
        "nll": float(-np.mean(logp[np.arange(len(gold)), gold])),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def stage1(
    model: PPRMoE,
    data: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    query = torch.from_numpy(data["query_embeddings"]).to(device)
    knowledge = torch.from_numpy(data["candidate_embeddings"]).to(device)
    utility_proxy = torch.from_numpy(data["utility_proxy"]).to(device)
    utility_target = torch.from_numpy(data["utility_target"]).to(device)
    params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("projection") and not name.startswith("position")
    ]
    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    losses: list[float] = []
    rank_losses: list[float] = []
    balance_losses: list[float] = []
    model.train()
    for epoch in range(MAX_STAGE1_EPOCHS):
        scores, routing, chosen = model(
            query,
            knowledge,
            utility_proxy,
        )
        rank_loss = model.rank_loss(scores, utility_target)
        balance_loss = model.balance_loss(routing, chosen)
        total = rank_loss + 0.01 * balance_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        losses.append(float(total.detach().cpu()))
        rank_losses.append(float(rank_loss.detach().cpu()))
        balance_losses.append(float(balance_loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 500 == 0:
            print(
                f"stage1 epoch={epoch + 1} loss={losses[-1]:.6f} "
                f"rank={rank_losses[-1]:.6f}",
                flush=True,
            )
        summary = convergence(
            losses,
            minimum=STAGE1_MIN_EPOCHS,
            window=STAGE1_WINDOW,
            relative_tolerance=0.01,
            absolute_tolerance=1e-4,
        )
        if summary["converged"]:
            break
    result = {
        **summary,
        "rank_final": rank_losses[-1],
        "balance_final": balance_losses[-1],
        "loss_history": losses,
        "rank_history": rank_losses,
        "balance_history": balance_losses,
    }
    return result


def stage2(
    model: PPRMoE,
    reader: FrozenReader,
    rows: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    query = torch.from_numpy(data["query_embeddings"]).to(device)
    knowledge = torch.from_numpy(data["candidate_embeddings"]).to(device)
    utility_proxy = torch.from_numpy(data["utility_proxy"]).to(device)
    utility_target = torch.from_numpy(data["utility_target"]).to(device)
    gold = torch.from_numpy(data["gold"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    losses: list[float] = []
    ce_losses: list[float] = []
    rank_losses: list[float] = []
    balance_losses: list[float] = []
    model.train()
    for epoch in range(STAGE2_MAX_EPOCHS):
        order = np.arange(len(rows))
        epoch_values = [[], [], [], []]
        for start in range(0, len(rows), batch_size):
            indices = order[start : start + batch_size]
            index_tensor = torch.from_numpy(indices).to(device)
            q_batch = query[index_tensor]
            z_batch = knowledge[index_tensor]
            proxy_batch = utility_proxy[index_tensor]
            target_batch = utility_target[index_tensor]
            gold_batch = gold[index_tensor]
            scores, routing, chosen = model(
                q_batch,
                z_batch,
                proxy_batch,
            )
            ranks = scores.detach().topk(TOP_CONTEXT, dim=-1).indices
            prefixes = model.prefix_embeddings(
                z_batch,
                proxy_batch,
                ranks,
            )
            tasks = [(int(index), []) for index in indices]
            logits = reader.logits(
                rows,
                tasks,
                prefixes=prefixes,
                require_grad=True,
            )
            ce_loss = F.cross_entropy(logits.float(), gold_batch)
            rank_loss = model.rank_loss(scores, target_batch)
            balance_loss = model.balance_loss(routing, chosen)
            total = ce_loss + rank_loss + 0.01 * balance_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_values[0].append(float(total.detach().cpu()))
            epoch_values[1].append(float(ce_loss.detach().cpu()))
            epoch_values[2].append(float(rank_loss.detach().cpu()))
            epoch_values[3].append(float(balance_loss.detach().cpu()))
        losses.append(float(np.mean(epoch_values[0])))
        ce_losses.append(float(np.mean(epoch_values[1])))
        rank_losses.append(float(np.mean(epoch_values[2])))
        balance_losses.append(float(np.mean(epoch_values[3])))
        print(
            f"stage2 epoch={epoch + 1} loss={losses[-1]:.6f} "
            f"ce={ce_losses[-1]:.6f} rank={rank_losses[-1]:.6f}",
            flush=True,
        )
        summary = convergence(
            losses,
            minimum=STAGE2_MIN_EPOCHS,
            window=STAGE2_WINDOW,
            relative_tolerance=0.01,
            absolute_tolerance=0.005,
        )
        if summary["converged"]:
            break
    return {
        **summary,
        "ce_final": ce_losses[-1],
        "rank_final": rank_losses[-1],
        "balance_final": balance_losses[-1],
        "loss_history": losses,
        "ce_history": ce_losses,
        "rank_history": rank_losses,
        "balance_history": balance_losses,
    }


def checkpoint_path() -> Path:
    return ARTIFACTS / READER_KEY / "ppr_moe.pt"


def train(
    device_name: str,
    batch_size: int,
) -> None:
    device = torch.device(device_name)
    data = load_assets("train")
    reader = FrozenReader(device_name)
    rows = load_rows("train")
    model = PPRMoE(reader.hidden_size).to(device)
    stage1_result = stage1(model, data, device)
    print(
        f"stage1 converged={stage1_result['converged']} "
        f"epochs={stage1_result['epochs']}",
        flush=True,
    )
    if not stage1_result["converged"]:
        raise RuntimeError("PPR MoE stage 1 did not converge")
    stage2_result = stage2(
        model,
        reader,
        rows,
        data,
        device,
        batch_size,
    )
    print(
        f"stage2 converged={stage2_result['converged']} "
        f"epochs={stage2_result['epochs']}",
        flush=True,
    )
    if not stage2_result["converged"]:
        raise RuntimeError("PPR MoE stage 2 did not converge")
    output = checkpoint_path()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "reader": READER_NAME,
            "reader_model_dir": str(MEDMO8B_DIR),
            "reader_dim": reader.hidden_size,
            "experts": 6,
            "top_experts": 2,
            "candidate_features": "q, z_i, utility_proxy",
            "utility_train_target": "gold-conditioned counterfactual margin",
            "utility_inference": "no-RAG-prediction counterfactual margin",
            "prefix_position_encoding": "preserved reader-native multimodal MRoPE",
            "stage1": stage1_result,
            "stage2": stage2_result,
        },
        output,
    )
    write_json(
        output.with_suffix(".json"),
        {
            "reader": READER_NAME,
            "reader_model_dir": str(MEDMO8B_DIR),
            "checkpoint": str(output),
            "stage1": {
                key: value
                for key, value in stage1_result.items()
                if key != "loss_history"
                and key != "rank_history"
                and key != "balance_history"
            },
            "stage2": {
                key: value
                for key, value in stage2_result.items()
                if key != "loss_history"
                and key != "ce_history"
                and key != "rank_history"
                and key != "balance_history"
            },
        },
    )
    reader.close()


def predict(
    device_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device(device_name)
    data = load_assets("test")
    checkpoint = torch.load(
        checkpoint_path(),
        map_location=device,
        weights_only=True,
    )
    if not checkpoint["stage1"]["converged"] or not checkpoint["stage2"]["converged"]:
        raise RuntimeError("PPR MoE checkpoint is not fully converged")
    reader = FrozenReader(device_name)
    model = PPRMoE(int(checkpoint["reader_dim"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    query = torch.from_numpy(data["query_embeddings"]).to(device)
    knowledge = torch.from_numpy(data["candidate_embeddings"]).to(device)
    utility_proxy = torch.from_numpy(data["utility_proxy"]).to(device)
    with torch.no_grad():
        scores, routing, chosen = model(query, knowledge, utility_proxy)
        ranks = scores.topk(TOP_CONTEXT, dim=-1).indices
        prefixes = model.prefix_embeddings(
            knowledge,
            utility_proxy,
            ranks,
        )
    rows = load_rows("test")
    probabilities = np.empty((len(rows), 4), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        tasks = [(index, []) for index in range(start, end)]
        logits = reader.logits(
            rows,
            tasks,
            prefixes=prefixes[start:end],
            require_grad=False,
        )
        probabilities[start:end] = torch.softmax(
            logits.float(),
            dim=-1,
        ).cpu().numpy()
        if start == 0 or end == len(rows) or end % 100 == 0:
            print(f"{READER_NAME} PPR MoE: {end}/{len(rows)}", flush=True)
    output = {
        "scores": scores.cpu().numpy(),
        "ranks": ranks.cpu().numpy(),
        "routing": routing.cpu().numpy(),
        "chosen_experts": chosen.cpu().numpy(),
        "probabilities": probabilities,
    }
    reader.close()
    return probabilities, output


def evaluate(
    device_name: str,
    batch_size: int,
) -> None:
    probabilities, output = predict(device_name, batch_size)
    data = load_assets("test")
    rows = load_rows("test")
    gold = answer_indices(rows)
    metrics = {
        "method": "PBridge",
        "reader": READER_NAME,
        "reader_model_dir": str(MEDMO8B_DIR),
        "dataset": "PediatricsMQA fixed 8:2 split",
        "split": "test",
        "metrics": evaluate_probabilities(probabilities, gold),
        "metadata": {
            "checkpoint": str(checkpoint_path()),
            "gme_candidates": 20,
            "experts": 6,
            "top_experts_per_knowledge": 2,
            "expert_form": "Z W_e q",
            "router_input": "concat(q, z_i, utility_proxy_i)",
            "expert_input_dim": 1536,
            "reader_prefix_tokens": 3,
            "prefix_position_encoding": "preserved reader-native multimodal MRoPE",
            "reader_hidden_size": int(
                torch.load(
                    checkpoint_path(),
                    map_location="cpu",
                    weights_only=True,
                )["reader_dim"]
            ),
            "utility_inference": (
                "counterfactual margin relative to the no-RAG predicted "
                "option; no test answer used"
            ),
            "training_converged": True,
        },
    }
    out_dir = ARTIFACTS / READER_KEY
    write_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "predictions.npz", **output)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        predictions = probabilities.argmax(axis=1)
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "gold": LABELS[int(gold[index])],
                        "prediction": LABELS[int(predictions[index])],
                        "correct": bool(predictions[index] == gold[index]),
                        "selected_candidate_ranks": output["ranks"][index].tolist(),
                        "selected_knowledge_indices": data[
                            "candidate_indices"
                        ][index, output["ranks"][index]].tolist(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if args.command == "train":
        train(args.device, args.batch_size)
    else:
        evaluate(args.device, args.batch_size)


if __name__ == "__main__":
    main()
