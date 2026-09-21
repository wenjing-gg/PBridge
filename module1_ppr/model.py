"""Prior-aware reranking MoE used by PBridge."""

from __future__ import annotations

import torch
from torch import nn


EXPERTS = 6
TOP_EXPERTS = 2
GME_DIM = 1536
ROUTER_HIDDEN = 1024
TOP_K = 20
TOP_CONTEXT = 3


class PPRMoE(nn.Module):
    def __init__(self, reader_dim: int):
        super().__init__()
        self.reader_dim = int(reader_dim)
        self.router = nn.Sequential(
            nn.Linear(GME_DIM + GME_DIM + 1, ROUTER_HIDDEN),
            nn.GELU(),
            nn.Linear(ROUTER_HIDDEN, EXPERTS),
        )
        self.expert_matrices = nn.Parameter(
            torch.empty(EXPERTS, GME_DIM, GME_DIM)
        )
        self.projection = nn.Linear(GME_DIM, reader_dim, bias=False)
        self.position = nn.Parameter(torch.empty(TOP_CONTEXT, GME_DIM))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.expert_matrices, mean=0.0, std=0.02)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(
        self,
        query: torch.Tensor,
        knowledge: torch.Tensor,
        utility_proxy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.float()
        knowledge = knowledge.float()
        utility_proxy = utility_proxy.float()
        expert_scores = torch.einsum(
            "bkd,edh,bh->bke",
            knowledge,
            self.expert_matrices,
            query,
        )
        router_input = torch.cat(
            [
                query[:, None, :].expand_as(knowledge),
                knowledge,
                utility_proxy.unsqueeze(-1),
            ],
            dim=-1,
        )
        router_logits = self.router(router_input)
        chosen = router_logits.topk(TOP_EXPERTS, dim=-1).indices
        selected_router = torch.gather(router_logits, -1, chosen)
        gates = torch.softmax(selected_router, dim=-1)
        selected_scores = torch.gather(expert_scores, -1, chosen)
        scores = (selected_scores * gates).sum(dim=-1)
        routing_probs = torch.softmax(router_logits, dim=-1)
        return scores, routing_probs, chosen

    def balance_loss(
        self,
        routing_probs: torch.Tensor,
        chosen: torch.Tensor,
    ) -> torch.Tensor:
        selected = torch.zeros_like(routing_probs)
        selected.scatter_(-1, chosen, 1.0)
        frequency = selected.mean(dim=(0, 1))
        probability = routing_probs.mean(dim=(0, 1))
        return EXPERTS * torch.sum(frequency * probability)

    def rank_loss(
        self,
        scores: torch.Tensor,
        utility_target: torch.Tensor,
    ) -> torch.Tensor:
        utility_diff = utility_target[:, :, None] - utility_target[:, None, :]
        score_diff = scores[:, :, None] - scores[:, None, :]
        valid = utility_diff > 0
        if not bool(valid.any()):
            return scores.sum() * 0.0
        margin = torch.tanh(utility_diff[valid])
        return torch.nn.functional.softplus(
            -(score_diff[valid] - margin)
        ).mean()

    def prefix_embeddings(
        self,
        knowledge: torch.Tensor,
        predicted_scores: torch.Tensor,
        ranks: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        selected_z = torch.gather(
            knowledge,
            1,
            ranks.unsqueeze(-1).expand(-1, -1, knowledge.shape[-1]),
        ).float()
        selected_scores = torch.gather(predicted_scores, 1, ranks).float()
        alpha = torch.softmax(selected_scores / temperature, dim=-1)
        positioned = (
            selected_z * alpha.unsqueeze(-1)
            + self.position.unsqueeze(0)
        )
        return self.projection(positioned)
