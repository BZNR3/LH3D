from __future__ import annotations
from typing import List, Optional, Dict
import os, random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader

from active_learning.methods import get_method


# ---------------------
# Utilities to count GTs robustly via *training-style* batches
# ---------------------

def _count_boxes_any(x) -> int:
    """Recursively count boxes in x.
    Treat a tensor/ndarray as boxes only if it's (N, D>=6).
    Skip any (..., 4, 4) camera/geom matrices.
    Also handles mmdet3d box objects with `.tensor`.
    """
    import torch
    if x is None:
        return 0
    if hasattr(x, "tensor"):
        return _count_boxes_any(getattr(x, "tensor"))
    if torch.is_tensor(x):
        if x.ndim >= 2 and tuple(x.shape[-2:]) == (4, 4):
            return 0
        if x.ndim >= 2 and x.shape[-1] >= 6:
            return int(x.reshape(-1, x.shape[-1]).shape[0])
        return 0
    if isinstance(x, np.ndarray):
        if x.ndim >= 2 and tuple(x.shape[-2:]) == (4, 4):
            return 0
        if x.ndim >= 2 and x.shape[-1] >= 6:
            return int(x.reshape(-1, x.shape[-1]).shape[0])
        return 0
    if isinstance(x, (list, tuple)):
        return sum(_count_boxes_any(t) for t in x)
    if isinstance(x, dict):
        return sum(_count_boxes_any(v) for v in x.values())
    return 0


def _count_labels_any(x) -> int:
    """Recursively count labels: one-dim (N,) tensors/arrays/lists.
    Skip any (...,4,4) matrix-like tensors.
    """
    import torch
    if x is None:
        return 0
    if torch.is_tensor(x):
        if x.ndim >= 2 and tuple(x.shape[-2:]) == (4, 4):
            return 0
        if x.ndim == 1:
            return int(x.shape[0])
        return 0
    if isinstance(x, np.ndarray):
        if x.ndim >= 2 and tuple(x.shape[-2:]) == (4, 4):
            return 0
        if x.ndim == 1:
            return int(x.shape[0])
        return 0
    if isinstance(x, (list, tuple)):
        return sum(_count_labels_any(t) for t in x)
    if isinstance(x, dict):
        return sum(_count_labels_any(v) for v in x.values())
    return 0


def _count_from_batch_tuple(batch) -> int:
    """Count total objects in a *training-style* batch.
    Expects: (sweep_imgs, mats, _, _, gt_boxes, gt_labels)
    """
    _, _, _, _, gt_boxes, gt_labels = batch
    total = 0
    if isinstance(gt_boxes, (list, tuple)):
        for b in gt_boxes:
            total += _count_boxes_any(b)
    else:
        total += _count_boxes_any(gt_boxes)
    if total == 0:
        if isinstance(gt_labels, (list, tuple)):
            for l in gt_labels:
                total += _count_labels_any(l)
        else:
            total += _count_labels_any(gt_labels)
    return int(total)


class ActiveLearner:
    """Round-based pool-based active learning orchestrator.

    Orchestrates: init label set -> train -> query -> object-budget check -> repeat.
    Compatible with BEVHeightLightningModel.
    """
    def __init__(self, model_cls, args, checkpoint_dir: Optional[str] = None):
        self.model_cls = model_cls
        self.args = args
        self.checkpoint_dir = checkpoint_dir or os.path.join(args.default_root_dir, "checkpoints_al")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Single model instance; weights persist across rounds
        self.model = model_cls(**vars(args)).to(self.device)

        # Build full train dataset once; model exposes a builder used by train_dataloader
        self.full_train_dataset = self.model._build_train_dataset()
        self.N = len(self.full_train_dataset)

        # Pools
        self._init_pools()

        # Active method
        Method = get_method(args.al_method)
        self.selector = Method()

        # Object budget
        self.obj_budget: Optional[int] = getattr(args, 'al_max_objects', None)
        self.obj_overshoot: int = int(getattr(args, 'al_objects_overshoot', 0))
        self._gt_count_cache: Dict[int, int] = {}
        self._budget_exhausted = False

        # [SEED-OFFSET] whether to exclude seed objects from budget
        self.ignore_seed_in_budget: bool = bool(getattr(args, 'al_ignore_seed_in_budget', True))
        self._seed_indices_snapshot: Optional[List[int]] = None
        self._seed_obj_offset: int = 0

        # [SEED-OFFSET] Take snapshot right after pools are initialized
        if self.ignore_seed_in_budget:
            self._seed_indices_snapshot = list(self.labeled_indices)
            self._seed_obj_offset = self._total_objects(self._seed_indices_snapshot) if self._seed_indices_snapshot else 0
            print(f"[AL] Seed snapshot: images={len(self._seed_indices_snapshot)} | "
                  f"objects={self._seed_obj_offset} (excluded from budget)")

    # ---------------------
    # Pools
    # ---------------------
    def _init_pools(self):
        rng = random.Random(self.args.al_pool_seed)
        all_idx = list(range(self.N))
        rng.shuffle(all_idx)
        init_sz = min(self.args.al_init_size, self.N)
        self.labeled_indices = sorted(all_idx[:init_sz])
        self.unlabeled_indices = sorted(all_idx[init_sz:])
        print(f"[AL] Dataset size={self.N} | init labeled={len(self.labeled_indices)} | pool={len(self.unlabeled_indices)}")

    # ---------------------
    # Trainers
    # ---------------------
    def _make_trainer(self, round_id: int) -> pl.Trainer:
        ckpt_dir = os.path.join(self.checkpoint_dir, f"round_{round_id}")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}",
            every_n_epochs=max(1, self.args.al_epochs_per_round // 2),
            save_last=True, save_top_k=-1,
        )
        trainer = pl.Trainer.from_argparse_args(
            self.args,
            max_epochs=self.args.al_epochs_per_round,
            callbacks=[checkpoint_callback],
        )
        return trainer

    # ---------------------
    # Object counting using training-style batches
    # ---------------------
    def _gt_count_for_index(self, idx: int) -> int:
        if idx in self._gt_count_cache:
            return self._gt_count_cache[idx]
        # Use a single-sample DataLoader to get a consistent collated batch
        subset = Subset(self.full_train_dataset, [idx])
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=self.model._collate_fn_for_train(),
        )
        batch = next(iter(loader))
        cnt = _count_from_batch_tuple(batch)
        self._gt_count_cache[idx] = int(cnt)
        return int(cnt)

    def _total_objects(self, indices: List[int]) -> int:
        if not indices:
            return 0
        subset = Subset(self.full_train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.model.batch_size_per_device,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            collate_fn=self.model._collate_fn_for_train(),
        )
        total = 0
        for batch in loader:
            total += _count_from_batch_tuple(batch)
        return int(total)

    # [SEED-OFFSET]
    def _spent_objects(self) -> int:
        """Objects that count against the budget (exclude seed if enabled)."""
        total = self._total_objects(self.labeled_indices)
        if not self.ignore_seed_in_budget:
            return total
        return max(0, total - self._seed_obj_offset)

    # [SEED-OFFSET] version of cap function
    def _cap_by_object_budget(self, picked_in_order: List[int]) -> List[int]:
        """Hard cap AFTER whole-image annotation.
        Add images following method order until crossing the cap; keep the image that crosses, then stop.
        Uses *spent* objects (excluding seed) for comparisons.
        """
        if not self.obj_budget:
            return picked_in_order

        cap = int(self.obj_budget)
        current_spent = self._spent_objects()

        kept: List[int] = []
        added_spent = 0
        for idx in picked_in_order:
            c = self._gt_count_for_index(idx)
            kept.append(idx)
            added_spent += c
            new_spent = current_spent + added_spent
            if new_spent > cap:
                print(f"[AL] Reached cap (excluding seed) after this image: {new_spent} > {cap}. Stop querying this round.")
                break
        return kept

    # ---------------------
    # Train / Query
    # ---------------------
    def _train_one_round(self, round_id: int):
        self.model.set_labeled_indices(self.labeled_indices)
        self._gt_count_cache.clear()  # reset per round
        trainer = self._make_trainer(round_id)
        try:
            from utils.backup_files import backup_codebase
            backup_codebase(os.path.join(self.args.default_root_dir, f'backup_round_{round_id}'))
        except Exception:
            pass
        trainer.fit(self.model)

    @torch.no_grad()
    def _query_unlabeled(self, k: int) -> List[int]:
        if k <= 0 or len(self.unlabeled_indices) == 0:
            return []
        subset = Subset(self.full_train_dataset, self.unlabeled_indices)
        loader = DataLoader(
            subset,
            batch_size=self.args.batch_size_per_device,
            shuffle=False,
            num_workers=4,
            collate_fn=self.model._collate_fn_for_train(),
            drop_last=False,
        )
        chosen_global_idx = self.selector.select(
            model=self.model,
            dataloader=loader,
            device=self.device,
            unlabeled_indices=self.unlabeled_indices,
            k=k,
        )
        return chosen_global_idx

    def run(self):
        R = int(self.args.al_rounds)
        Q = int(self.args.al_query_size)

        for r in range(R):
            print(f"[AL] ===== Round {r+1}/{R}: Train on {len(self.labeled_indices)} labeled samples =====")
            self._train_one_round(r)

            # Last round: do not query further (keep epochs constant)
            if r == R - 1:
                continue

            # [SEED-OFFSET] Budget check with 'spent' (exclude seed)
            if self.obj_budget:
                spent = self._spent_objects()
                print(f"[AL] Labeled objects so far (excluding seed={self._seed_obj_offset}): {spent} (cap={self.obj_budget})")
                if spent >= self.obj_budget:
                    print("[AL] Object cap reached (excluding seed). Will SKIP further querying but continue training.")
                    continue  # skip query, next round still trains on same set

            if len(self.unlabeled_indices) == 0:
                print("[AL] Pool empty. Skipping query and continuing training.")
                continue

            k = min(Q, len(self.unlabeled_indices))
            print(f"[AL] Querying top-{k} samples from pool={len(self.unlabeled_indices)}…")
            picked = self._query_unlabeled(k)
            picked = self._cap_by_object_budget(picked)

            # Move samples from pool -> labeled
            pool_set = set(self.unlabeled_indices)
            picked_set = set(picked)
            self.labeled_indices = sorted(set(self.labeled_indices).union(picked_set))
            self.unlabeled_indices = sorted(list(pool_set - picked_set))

            # Log (both total & spent)
            tot_objs = self._total_objects(self.labeled_indices)
            spent = self._spent_objects()
            print(f"[AL] After round {r + 1}: labeled_images={len(self.labeled_indices)}, "
                  f"labeled_objects_total={tot_objs}, spent(excl. seed)={spent}/{self.obj_budget if self.obj_budget else '∞'}")

        print("[AL] Finished all rounds.")
