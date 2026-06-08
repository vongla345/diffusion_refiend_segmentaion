from typing import Callable, Dict

import numpy as np
import torch


def compute_dice_np(pred, target, smooth=1e-6):
    pred = (pred > 0.5).astype(np.float32).ravel()
    target = target.ravel().astype(np.float32)
    inter = (pred * target).sum()
    return float((2 * inter + smooth) / (pred.sum() + target.sum() + smooth))


def compute_iou_np(pred, target, smooth=1e-6):
    pred = (pred > 0.5).astype(np.float32).ravel()
    target = target.ravel().astype(np.float32)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter + smooth) / (union + smooth))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    get_text_emb: Callable[[int], torch.Tensor],
) -> Dict[str, float]:
    # Intentionally keep model in train() mode instead of calling model.eval().
    # With very few labeled samples (e.g. ≤16), BatchNorm running statistics are
    # unreliable and produce extreme variance in val metrics epoch-to-epoch.
    # train() mode uses fresh batch statistics, which are far more stable.
    # @torch.no_grad() above already disables gradient computation.
    was_training = model.training
    model.train()
    dice_scores, iou_scores = [], []
    for batch in loader:
        img = batch["image_a"].to(device)
        mask = batch["mask"].cpu().numpy()
        txt = get_text_emb(img.shape[0])
        logits, _ = model(img, txt)
        probs = torch.sigmoid(logits).cpu().numpy()
        for p, t in zip(probs, mask):
            dice_scores.append(compute_dice_np(p[0], t[0]))
            iou_scores.append(compute_iou_np(p[0], t[0]))
    if not was_training:
        model.eval()
    return {
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "std": float(np.std(dice_scores)),
    }
