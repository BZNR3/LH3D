from __future__ import annotations
from typing import List
import numpy as np
import torch
from active_learning.base import BaseActiveSelector

class EntropySelector(BaseActiveSelector):
    """Object-class logit entropy (averaged per image).

    For each image, compute entropy over class probabilities per detection and
    average across detections. Higher average entropy => more uncertain image.

    Implementation tries two paths:
      (A) If the model exposes `get_class_logits_for_dets(preds, img_metas)`
          returning a list of (N_i, C) arrays per image, we use that to compute
          true multi-class entropy (softmax -> normalized Shannon entropy).
      (B) Otherwise, we fall back to *Bernoulli* entropy on detection score `p`
          from decoded bboxes (`get_bboxes`) as a proxy (normalized by ln 2).

    Images with no detections are assigned uncertainty 1.0.
    """

    @torch.no_grad()
    def score_unlabeled(self, model: torch.nn.Module, dataloader, device) -> List[float]:
        model.eval()
        scores_per_image: List[float] = []

        for batch in dataloader:
            sweep_imgs, mats, _, img_metas, _, _ = batch
            if torch.cuda.is_available():
                for k, v in mats.items():
                    mats[k] = v.to(device, non_blocking=True)
                sweep_imgs = sweep_imgs.to(device, non_blocking=True)

            # forward
            if next(model.model.parameters()).device != device:
                model.model.to(device)

            preds = model.model(sweep_imgs, mats)

            # Try path (A): per-detection class logits
            class_logits_list = None
            if hasattr(model.model, 'get_class_logits_for_dets'):
                try:
                    class_logits_list = model.model.get_class_logits_for_dets(preds, img_metas)
                except Exception:
                    class_logits_list = None

            if class_logits_list is not None:
                # Use multi-class entropy
                for logits in class_logits_list:
                    if logits is None or (isinstance(logits, (list, tuple)) and len(logits) == 0):
                        scores_per_image.append(1.0)
                        continue
                    logits = torch.as_tensor(logits).detach().cpu().float()
                    # softmax -> probs
                    probs = torch.softmax(logits, dim=-1).clamp_(1e-8, 1.0)
                    ent = (-probs * probs.log()).sum(dim=-1)
                    # normalize by ln(C)
                    C = probs.shape[-1]
                    ent = ent / np.log(C) if C > 1 else ent
                    val = float(ent.mean().item()) if ent.numel() > 0 else 1.0
                    scores_per_image.append(val)
                continue  # go to next batch

            # Path (B): fallback Bernoulli entropy using decoded scores
            if isinstance(model.model, torch.nn.parallel.DistributedDataParallel):
                dets = model.model.module.get_bboxes(preds, img_metas)
            else:
                dets = model.model.get_bboxes(preds, img_metas)

            for i in range(len(dets)):
                _, scores, _ = dets[i][:3]
                p = scores.detach().cpu().float().clamp_(1e-8, 1-1e-8)
                if p.numel() == 0:
                    scores_per_image.append(1.0)
                    continue
                ent = -(p * p.log() + (1 - p) * (1 - p).log())
                ent = ent / np.log(2.0)  # normalize to [0,1]
                scores_per_image.append(float(ent.mean().item()))

        return scores_per_image
