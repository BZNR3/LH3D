# =============================
# File: active_learning/base.py
# =============================
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import torch

class BaseActiveSelector:
    """Interface for active learning selection methods.

    Subclasses should implement `score_unlabeled` and may override `select`.
    `score_unlabeled` must return a list of floats aligned with the *order* of
    the provided `unlabeled_indices` (i.e., dataloader.dataset indices).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @torch.no_grad()
    def score_unlabeled(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> List[float]:
        raise NotImplementedError

    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
    ) -> List[int]:
        scores = self.score_unlabeled(model, dataloader, device)
        assert len(scores) == len(unlabeled_indices), (
            f"scores ({len(scores)}) must align with unlabeled_indices ({len(unlabeled_indices)})"
        )
        # Higher score = more uncertain
        topk_rel = torch.topk(torch.tensor(scores), k=min(k, len(scores))).indices.tolist()
        chosen = [unlabeled_indices[i] for i in topk_rel]
        return chosen

