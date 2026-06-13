from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


def soft_skel(img, iters=10):
    img1 = soft_dilate(soft_erode(img))
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_dilate(soft_erode(img))
        delta = F.relu(img - img1)
        skel = skel + delta - skel * delta
    return skel


class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, logits, targets):
        dilated = self.pool(targets)
        eroded = 1.0 - self.pool(1.0 - targets)
        boundary = (dilated - eroded).clamp(0, 1)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (bce_loss * boundary).sum()/(boundary.sum() + 1e-6)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        lambda_bce=1.0,
        lambda_dice=1.0,
        lambda_cldice=0.5,
        lambda_bound=0.5,
        cldice_iters=10,
    ):
        super().__init__()
        self.boundary = BoundaryLoss()
        self.lb = lambda_bce
        self.ld = lambda_dice
        self.lc = lambda_cldice
        self.lbo = lambda_bound
        self.cldice_iters = cldice_iters

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce    = F.binary_cross_entropy_with_logits(logits, targets)
        inter  = (probs * targets).sum()
        dice   = 1 - (2 * inter + 1) / (probs.sum() + targets.sum() + 1)
        skel_p = soft_skel(probs, self.cldice_iters)
        skel_t = soft_skel(targets, self.cldice_iters)
        tprec  = (skel_p * targets).sum() / (skel_p.sum() + 1e-6)
        tsens  = (skel_t * probs).sum()  / (skel_t.sum() + 1e-6)
        cldice = 1 - (2 * tprec * tsens) / (tprec + tsens + 1e-6)
        bnd    = self.boundary(logits, targets)
        total  = self.lb * bce + self.ld * dice + self.lc * cldice + self.lbo * bnd
        # Second element: raw (unweighted) component values for logging/plotting
        components = (bce.item(), dice.item(), cldice.item(), bnd.item())
        return total, components


def view_consistency_loss(logits_a, logits_b):
    """Cross-view consistency in probability space (bounded, >= 0).

    Pulls the strong view (``logits_b``, student) toward the weak view
    (``logits_a``, teacher) with BCE + Dice on probabilities. The weak view is
    detached so it acts as a fixed soft target (FixMatch / Mean-Teacher style),
    which keeps the loss bounded and stops the two views from inflating their
    logits together.

    NOTE: the previous version fed raw logits as the BCE target and computed
    Dice on raw logits — both unbounded below — which drove a logit-inflation
    collapse once this loss actually backpropagated (negative ``loss_view``).
    """
    target = torch.sigmoid(logits_a).detach()
    bce = F.binary_cross_entropy_with_logits(logits_b, target)
    prob = torch.sigmoid(logits_b)
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


def compute_masked_pseudo_loss(
    logits, pseudo_mask, valid_mask
) -> Optional[torch.Tensor]:
    valid = valid_mask > 0.5
    if valid.sum() == 0:
        return None
    lv = logits[valid]
    tv = pseudo_mask[valid]
    bce = F.binary_cross_entropy_with_logits(lv, tv)
    prob = torch.sigmoid(lv)
    inter = (prob * tv).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + tv.sum() + 1)
    return bce + dice


def soft_consistency_loss(
    logits,
    target_prob,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """CorrMatch-style soft supervision (Eq. 3 of the CorrMatch paper).

    Instead of hardening the teacher prediction to a 0/1 pseudo-label and
    discarding the distribution information, this matches the *soft* teacher
    distribution with a per-pixel Kullback-Leibler divergence, restricted to
    high-confidence pixels (``valid_mask``).

    For binary segmentation every pixel is a Bernoulli distribution, so the KL
    reduces to the closed form below. The teacher (``target_prob``) is detached
    so gradients only update the student (``logits``):

        KL(teacher || student)
          = q*log(q/p) + (1-q)*log((1-q)/(1-p))

    where ``p = sigmoid(logits)`` (student) and ``q = target_prob`` (teacher).

    Args:
        logits:      student logits, shape (B, 1, H, W).
        target_prob: teacher probabilities in [0, 1], same shape as ``logits``.
        valid_mask:  optional high-confidence mask (B, 1, H, W); when given the
                     KL is averaged over valid pixels only. Returns ``None`` if
                     no pixel is valid.
    """
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    q = target_prob.detach().clamp(eps, 1.0 - eps)
    kl = q * (torch.log(q) - torch.log(p)) + (1 - q) * (
        torch.log(1 - q) - torch.log(1 - p)
    )
    if valid_mask is not None:
        valid = valid_mask > 0.5
        if valid.sum() == 0:
            return None
        return kl[valid].mean()
    return kl.mean()
