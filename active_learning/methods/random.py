from __future__ import annotations
from typing import List
import random
import torch
from active_learning.base import BaseActiveSelector

class RandomSelector(BaseActiveSelector):
    """Uniform random selection from the unlabeled pool.

    Ignores the model/dataloader and just samples K indices at random.
    Deterministic given `seed` in kwargs (default=0).
    """
    def select(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
    ) -> List[int]:
        if k <= 0 or len(unlabeled_indices) == 0:
            return []
        seed = int(self.kwargs.get("seed", 0))
        rng = random.Random(seed)
        pool = list(unlabeled_indices)
        rng.shuffle(pool)
        return pool[:min(k, len(pool))]