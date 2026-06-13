import copy
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from datasets.dataloader import DiffusionDataset, LabeledDataset, UnlabeledDataset, resolve_split_dirs
from evaluation.metrics import evaluate
from losses.segmentation_loss import (
    SegmentationLoss,
    compute_masked_pseudo_loss,
    soft_consistency_loss,
    view_consistency_loss,
)
from models.diffusion import DiffusionScheduler, MiniUNet
from models.backbone import SegmentationModel, build_segmentation_model

logger = logging.getLogger(__name__)


@torch.no_grad()
def _ema_update(ema_model: torch.nn.Module, student: torch.nn.Module, alpha: float) -> None:
    """Exponential Moving Average weight update: ema = alpha*ema + (1-alpha)*student.

    Learnable parameters are EMA-averaged.
    BN running statistics (buffers) are copied directly from the student so
    that the teacher's normalisation stays consistent with training-time stats.
    """
    for p_ema, p_s in zip(ema_model.parameters(), student.parameters()):
        p_ema.data.mul_(alpha).add_(p_s.data, alpha=1.0 - alpha)
    for b_ema, b_s in zip(ema_model.buffers(), student.buffers()):
        b_ema.copy_(b_s)


def get_conf_schedule(
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    is_warmup: bool,
    conf_strict_epochs: int,
    keep_frac_min: float,
    keep_frac_max: float,
) -> Tuple[bool, float]:
    """Hybrid FixMatch-style confidence curriculum (strict early, then relax).

    Returns ``(strict, keep_frac)``:
      * strict=True  → for the first ``conf_strict_epochs`` semi-sup epochs the
        caller uses a fixed strict threshold (tau_max).
      * strict=False → percentile phase; ``keep_frac`` ramps linearly from
        ``keep_frac_min`` to ``keep_frac_max`` over the remaining semi-sup
        epochs (keep a growing fraction of the most confident pixels).
    """
    e0 = (warmup_epochs if is_warmup else 0) + 1   # first semi-sup epoch
    semi = epoch - e0                              # 0 on the first semi-sup epoch
    if semi < conf_strict_epochs:
        return True, keep_frac_min
    total = max(1, epochs - e0 - conf_strict_epochs)
    p2 = min(max((semi - conf_strict_epochs) / total, 0.0), 1.0)
    keep_frac = keep_frac_min + (keep_frac_max - keep_frac_min) * p2
    return False, keep_frac

def get_pseudo_weight(
    epoch: int,
    max_epochs: int,
    lambda_pseudo: float,
    warmup_epochs: int = 0,
) -> float:
    """Ramp pseudo-label weight from 0 to lambda_pseudo over the semi-sup phase.

    The warmup phase (epochs 1..warmup_epochs) counts as epoch 0 — the weight
    starts growing only once semi-supervised training begins.
    """
    semi_sup_epoch = max(0, epoch - warmup_epochs)  # 0 during warmup phase
    if semi_sup_epoch == 0:
        return 0.0
    semi_sup_total = max(1, max_epochs - warmup_epochs)
    rampup_length = semi_sup_total // 2
    if semi_sup_epoch >= rampup_length:
        return lambda_pseudo
    return lambda_pseudo * (semi_sup_epoch / max(rampup_length, 1))


PSEUDO_ABLATION_MODES = ("strong_weak", "weak_weak", "strong_strong", "diffrect")


def resolve_pseudo_ablation(cfg: Dict[str, Any]) -> str:
    mode = str(cfg.get("train", {}).get("pseudo_ablation", "strong_weak")).lower()
    if mode not in PSEUDO_ABLATION_MODES:
        raise ValueError(
            f"train.pseudo_ablation must be one of {PSEUDO_ABLATION_MODES}, got {mode!r}"
        )
    return mode


CONSISTENCY_TYPES = ("traditional", "soft", "combined")


def resolve_consistency_type(cfg: Dict[str, Any]) -> str:
    """Which consistency loss to use for the unsupervised terms.

    Applies to BOTH ``loss_refined_unsup`` (pseudo-label loss) and
    ``loss_view_unsup`` (cross-view consistency):
      traditional – BCE + Dice (hard pseudo-label / view consistency; current).
      soft        – CorrMatch-style KL soft supervision on the soft teacher
                    distribution (no hard thresholding of the target).
      combined    – traditional + ``lambda_soft`` * soft.
    """
    mode = str(cfg.get("loss", {}).get("consistency_type", "traditional")).lower()
    if mode not in CONSISTENCY_TYPES:
        raise ValueError(
            f"loss.consistency_type must be one of {CONSISTENCY_TYPES}, got {mode!r}"
        )
    return mode


def _combine_consistency(
    trad: Optional[torch.Tensor],
    soft: Optional[torch.Tensor],
    lambda_soft: float,
) -> Optional[torch.Tensor]:
    """None-safe combination of the traditional and soft consistency terms.

    The non-None terms decide the effective mode:
      * only ``trad``        → traditional
      * only ``soft``        → soft (full weight; the caller's pseudo/view
                                weight still scales it)
      * both                 → combined: trad + lambda_soft * soft
    """
    if trad is None and soft is None:
        return None
    if soft is None:
        return trad
    if trad is None:
        return soft
    return (1 - lambda_soft) * trad + lambda_soft * soft


def refined_consistency_loss(
    logits: torch.Tensor,
    hard_pseudo: torch.Tensor,
    soft_pseudo: torch.Tensor,
    valid_mask: torch.Tensor,
    consistency_type: str,
    lambda_soft: float,
) -> Optional[torch.Tensor]:
    """Pseudo-label (refined) loss under the selected consistency type.

    traditional → masked BCE+Dice against the hard (0/1) pseudo-label.
    soft        → masked KL against the soft (refined) teacher distribution.
    combined    → both.
    """
    trad = soft = None
    if consistency_type in ("traditional", "combined"):
        trad = compute_masked_pseudo_loss(logits, hard_pseudo, valid_mask)
    if consistency_type in ("soft", "combined"):
        soft = soft_consistency_loss(logits, soft_pseudo, valid_mask)
    return _combine_consistency(trad, soft, lambda_soft)


def view_consistency_combined(
    logits_weak: torch.Tensor,
    logits_strong: torch.Tensor,
    consistency_type: str,
    lambda_soft: float,
) -> Optional[torch.Tensor]:
    """Cross-view consistency loss under the selected consistency type.

    traditional → BCE+Dice between the two views.
    soft        → KL matching the strong view to the (detached) weak view.
    combined    → both.
    """
    trad = soft = None
    if consistency_type in ("traditional", "combined"):
        trad = view_consistency_loss(logits_weak, logits_strong)
    if consistency_type in ("soft", "combined"):
        soft = soft_consistency_loss(logits_strong, torch.sigmoid(logits_weak))
    return _combine_consistency(trad, soft, lambda_soft)


def compute_diffusion_unsup_losses(
    seg_model: torch.nn.Module,
    diff_model: torch.nn.Module,
    diff_sched: DiffusionScheduler,
    img_ua: torch.Tensor,
    img_ub: torch.Tensor,
    txt_u: Optional[torch.Tensor],
    device: torch.device,
    get_text_emb: Callable[[int], torch.Tensor],
    noise_frac: float,
    strict: bool,
    keep_frac: float,
    tau_min: float,
    tau_max: float,
    pseudo_ablation: str,
    consistency_type: str = "traditional",
    lambda_soft: float = 0.25,
    teacher_model: Optional[torch.nn.Module] = None,
):
    """Diffusion-refined pseudo-label ablations for unlabeled loss.

    Modes (train.pseudo_ablation):
      strong_weak  – FixMatch: strong logits vs refined weak mask (default)
      weak_weak    – DiffRect L_Rect: weak logits vs refined weak mask
      strong_strong – self-training: strong logits vs refined strong mask
      diffrect     – weak_weak + strong_weak (both vs refined weak mask)

    consistency_type selects how both ``loss_refined_unsup`` and
    ``loss_view_unsup`` are computed: traditional (BCE+Dice), soft (CorrMatch
    KL on the soft teacher distribution), or combined (trad + lambda_soft*soft).

    teacher_model: if provided (EMA model), pseudo-labels are generated from the
    stable teacher instead of the current student. Student predictions
    (logits_ua, logits_ub) always come from seg_model.

    NOTE: pseudo-label generation and the confidence-mask computation run under
    ``torch.no_grad`` (they must not receive gradients), but the student forward
    passes and the returned losses DO build a graph so the unsupervised loss
    actually trains the model.
    """
    # ---- Pseudo-label target + confidence mask (no gradients) --------------
    with torch.no_grad():
        refine_weak = pseudo_ablation in ("strong_weak", "weak_weak", "diffrect")
        refine_img = img_ua if refine_weak else img_ub
        pseudo_source = teacher_model if teacher_model is not None else seg_model

        refined_pseudo, conf_map = generate_and_refine_pseudo_batch(
            pseudo_source,
            diff_model,
            diff_sched,
            refine_img,
            device,
            get_text_emb,
            noise_frac,
        )
        # Per-image confidence threshold (hybrid curriculum):
        #   strict phase     → fixed tau_max
        #   percentile phase → keep the top `keep_frac` most-confident pixels of
        #                      each image, clamped to [tau_min, tau_max].
        b = conf_map.shape[0]
        if strict:
            thr = torch.full((b,), tau_max, device=conf_map.device, dtype=torch.float32)
        else:
            flat = conf_map.reshape(b, -1).float()
            q = min(max(1.0 - keep_frac, 0.0), 1.0)
            thr = torch.quantile(flat, q, dim=1).clamp(tau_min, tau_max)
        valid_mask = (conf_map > thr.view(b, 1, 1, 1)).float()
        valid_px_ratio = valid_mask.mean().item()
        conf_thresh_eff = float(thr.mean().item())
        # Hard (0/1) target for traditional supervision; the soft target is the
        # raw refined probability map used by CorrMatch-style soft supervision.
        hard_pseudo = (refined_pseudo > 0.5).float()
        soft_pseudo = refined_pseudo

    if valid_mask.sum() == 0:
        if pseudo_ablation == "diffrect":
            return None, (None, None), None, valid_px_ratio, conf_thresh_eff, True
        return None, None, valid_px_ratio, conf_thresh_eff, True

    # ---- Student forward + unsupervised losses (with gradients) ------------
    logits_ua, _ = seg_model(img_ua, txt_u)
    logits_ub, _ = seg_model(img_ub, txt_u)

    if pseudo_ablation == "strong_weak":
        loss_refined_unsup = refined_consistency_loss(
            logits_ub, hard_pseudo, soft_pseudo, valid_mask, consistency_type, lambda_soft
        )
    elif pseudo_ablation == "weak_weak":
        loss_refined_unsup = refined_consistency_loss(
            logits_ua, hard_pseudo, soft_pseudo, valid_mask, consistency_type, lambda_soft
        )
    elif pseudo_ablation == "strong_strong":
        loss_refined_unsup = refined_consistency_loss(
            logits_ub, hard_pseudo, soft_pseudo, valid_mask, consistency_type, lambda_soft
        )
    elif pseudo_ablation == "diffrect":
        l_rect = refined_consistency_loss(
            logits_ua, hard_pseudo, soft_pseudo, valid_mask, consistency_type, lambda_soft
        )
        l_fix = refined_consistency_loss(
            logits_ub, hard_pseudo, soft_pseudo, valid_mask, consistency_type, lambda_soft
        )
        if l_rect is None and l_fix is None:
            loss_refined_unsup = None
        elif l_rect is None:
            loss_refined_unsup = l_fix
        elif l_fix is None:
            loss_refined_unsup = l_rect
        else:
            loss_refined_unsup = 0.5 * l_rect + 0.5 * l_fix
    else:
        raise ValueError(f"Unknown pseudo_ablation: {pseudo_ablation}")

    loss_view_unsup = view_consistency_combined(
        logits_ua, logits_ub, consistency_type, lambda_soft
    )

    if pseudo_ablation == "diffrect":
        return loss_refined_unsup, (l_rect, l_fix), loss_view_unsup, valid_px_ratio, conf_thresh_eff, loss_refined_unsup is None
    else:
        return loss_refined_unsup, loss_view_unsup, valid_px_ratio, conf_thresh_eff, loss_refined_unsup is None


@torch.no_grad()
def generate_and_refine_pseudo_batch(
    seg_model: torch.nn.Module,
    diff_model: torch.nn.Module,
    diff_scheduler: DiffusionScheduler,
    img_batch: torch.Tensor,
    device: torch.device,
    get_text_emb: Callable[[int], torch.Tensor],
    noise_frac: float = 0.4,
):
    seg_model.eval()
    b = img_batch.shape[0]
    txt = get_text_emb(b)  # may be None when use_text_encoder=false
    refined_masks, conf_maps = [], []

    for i in range(b):
        img_single = img_batch[i : i + 1]
        txt_slice = txt[i : i + 1] if txt is not None else None
        logits, _ = seg_model(img_single, txt_slice)
        pseudo_prob = torch.sigmoid(logits)

        x0 = pseudo_prob * 2 - 1
        t_start = int(diff_scheduler.T * noise_frac)
        noise = torch.randn_like(x0)
        ab_t = diff_scheduler.alphas_bar[t_start]
        xt = torch.sqrt(ab_t) * x0 + torch.sqrt(1 - ab_t) * noise

        for t in reversed(range(t_start)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            eps_pred = diff_model(xt, t_batch, img_single)
            ab_t = diff_scheduler.alphas_bar[t]
            alpha_t = ab_t / diff_scheduler.alphas_bar[t - 1] if t > 0 else ab_t
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - ab_t)
            mean = coef1 * (xt - coef2 * eps_pred)
            if t > 0:
                ab_prev = diff_scheduler.alphas_bar[t - 1]
                sigma = torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - alpha_t))
                xt = mean + sigma * torch.randn_like(xt)
            else:
                xt = mean

        refined = torch.clamp((xt + 1) / 2, 0, 1)
        conf = torch.maximum(refined, 1 - refined)
        refined_masks.append(refined)
        conf_maps.append(conf)

    seg_model.train()
    return torch.cat(refined_masks, dim=0), torch.cat(conf_maps, dim=0)


def train(
    cfg: Dict[str, Any],
    device: torch.device,
    get_text_emb: Callable[[int], torch.Tensor],
) -> Tuple[MiniUNet, SegmentationModel]:
    paths = cfg["paths"]
    data_root = paths["data_root"]
    ckpt_dir = paths["checkpoint_dir"]
    log_dir = paths.get("log_dir", ckpt_dir)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    diff_c = cfg["diffusion"]
    seg_c = cfg["segmentation"]
    loss_c = cfg["loss"]
    tr = cfg["train"]
    epochs = tr["epochs"]
    warmup_epochs = tr["warmup_epochs"]
    is_warmup = tr.get("is_warmup", True)
    use_diffusion = tr.get("use_diffusion", True)
    use_ema_teacher = bool(tr.get("use_ema_teacher", False))
    ema_alpha = float(tr.get("ema_alpha", 0.999))
    pseudo_ablation = resolve_pseudo_ablation(cfg)
    consistency_type = resolve_consistency_type(cfg)
    lambda_soft = float(loss_c.get("lambda_soft", 0.25))
    # Accepts true/false (YAML booleans) or the special string "partly".
    #   true     → encoder frozen for the whole training
    #   false    → encoder never frozen
    #   "partly" → frozen during warmup, unfrozen after (requires is_warmup: true)
    _fe = seg_c.get("freeze_enc", True)
    if _fe == "partly":
        freeze_enc_mode = "partly"
    elif _fe is True or str(_fe).lower() in ("yes", "true"):
        freeze_enc_mode = "yes"
    else:
        freeze_enc_mode = "no"

    if freeze_enc_mode == "partly" and not is_warmup:
        raise ValueError("freeze_enc='partly' requires is_warmup=True in the config")

    patch_cfg = cfg.get("patch") or {}
    train_im_size = int(patch_cfg.get("size", 256))
    crop_retry = int(patch_cfg.get("crop_retry", 1))
    min_fg_ratio = float(patch_cfg.get("min_fg_ratio", 0.0))
    accum_steps = max(1, int(seg_c.get("accum_steps", 1)))
    grad_ckpt = bool(seg_c.get("grad_checkpointing", False))

    resolved = resolve_split_dirs(data_root)
    missing = [k for k in ("labeled", "unlabeled", "val") if k not in resolved]
    if missing:
        raise FileNotFoundError(f"Missing splits in data_root={data_root}: {missing}")

    sup_img, sup_mask = resolved["labeled"]
    unsup_img, _ = resolved["unlabeled"]
    val_img, val_mask = resolved["val"]
    if not sup_mask or not val_mask:
        raise FileNotFoundError("labeled/val mask directories are required for training.")

    diff_ds = DiffusionDataset(
        sup_img,
        sup_mask,
        patch_size=train_im_size,
        crop_retry=crop_retry,
        min_fg_ratio=min_fg_ratio,
        is_train=True,
    )
    diff_val_ds = DiffusionDataset(
        val_img,
        val_mask,
        patch_size=train_im_size,
        crop_retry=1,
        min_fg_ratio=0.0,
        is_train=False,
    )
    sup_ds = LabeledDataset(
        sup_img,
        sup_mask,
        is_train=True,
        patch_size=train_im_size,
        crop_retry=crop_retry,
        min_fg_ratio=min_fg_ratio,
    )
    unsup_ds = UnlabeledDataset(
        unsup_img,
        patch_size=train_im_size,
    )
    val_ds = LabeledDataset(
        val_img,
        val_mask,
        is_train=False,
        patch_size=train_im_size,
        crop_retry=1,
        min_fg_ratio=0.0,
    )

    geom = (
        f"patches {train_im_size}x{train_im_size} "
        f"(train: pad+random crop reflect101; val: resize)"
    )
    logger.info(
        "Labeled: %d | Unlabeled: %d | Val: %d | %s | crop_retry=%d min_fg=%.4f | "
        "seg accum_steps=%d (~batch %d) | grad_ckpt=%s",
        len(sup_ds),
        len(unsup_ds),
        len(val_ds),
        geom,
        crop_retry,
        min_fg_ratio,
        accum_steps,
        seg_c["batch_size"] * accum_steps,
        grad_ckpt,
    )

    diff_ld = DataLoader(
        diff_ds,
        batch_size=diff_c["batch_size"],
        shuffle=True,
        num_workers=tr["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    diff_val_ld = DataLoader(
        diff_val_ds,
        batch_size=diff_c["batch_size"],
        shuffle=False,
        num_workers=tr["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    repeat = min(max(1, len(unsup_ds) // max(len(sup_ds), 1)), seg_c["max_repeat"])
    sup_ld = DataLoader(
        ConcatDataset([sup_ds] * repeat),
        batch_size=seg_c["batch_size"],
        shuffle=True,
        num_workers=tr["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    unsup_ld = DataLoader(
        unsup_ds,
        batch_size=seg_c["batch_size"],
        shuffle=True,
        num_workers=tr["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=seg_c["val_batch_size"],
        shuffle=False,
        num_workers=tr["num_workers"],
        pin_memory=True,
    )

    diff_model = MiniUNet(
        base_ch=diff_c["base_ch"], depth=diff_c.get("depth", 4), T=diff_c["T"]
    ).to(device)
    diff_sched = DiffusionScheduler(T=diff_c["T"], device=device)
    initial_freeze = freeze_enc_mode in ("partly", "yes")
    seg_model = build_segmentation_model(
        cfg, freeze_enc=initial_freeze, grad_checkpointing=grad_ckpt,
        use_pyramid_feature=seg_c["use_pyramid_feature"]
    ).to(device)
    logger.info("Segmentation decoder: deeplabv3")
    if use_diffusion:
        logger.info("Pseudo-label ablation: %s", pseudo_ablation)
    else:
        logger.info(
            "use_diffusion=false — unlabeled loss is view consistency only "
            "(no diffusion-refined pseudo-label loss)"
        )
    if consistency_type == "combined":
        logger.info("Consistency loss: %s (lambda_soft=%.4f)", consistency_type, lambda_soft)
    else:
        logger.info("Consistency loss: %s", consistency_type)

    if use_ema_teacher:
        ema_model = copy.deepcopy(seg_model)
        for p in ema_model.parameters():
            p.requires_grad = False  # teacher never receives gradients
        ema_model.train()            # train mode → BN uses batch stats (same reason as evaluate())
        logger.info("EMA teacher enabled (alpha=%.4f) — pseudo-labels from stable teacher", ema_alpha)
    else:
        ema_model = None
        logger.info("EMA teacher disabled — pseudo-labels from current student model")

    seg_loss_fn = SegmentationLoss(
        lambda_bce=loss_c.get("lambda_bce", 1.0),
        lambda_dice=loss_c.get("lambda_dice", 1.0),
        lambda_cldice=loss_c["lambda_cldice"],
        lambda_bound=loss_c["lambda_bound"],
        cldice_iters=int(loss_c.get("cldice_iters", 10)),
    )

    diff_opt = torch.optim.AdamW(diff_model.parameters(), lr=diff_c["lr"], weight_decay=diff_c["weight_decay"])
    diff_lrsched = torch.optim.lr_scheduler.CosineAnnealingLR(diff_opt, epochs, eta_min=1e-7)
    seg_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, seg_model.parameters()),
        lr=seg_c["lr"],
        weight_decay=seg_c['weight_decay'],
    )
    seg_lrsched = torch.optim.lr_scheduler.CosineAnnealingLR(seg_opt, epochs, eta_min=1e-7)
    use_amp = device.type == "cuda"
    seg_scaler = GradScaler() if use_amp else None

    best_iou = 0.0
    best_diff_loss = float("inf")
    no_improve = 0
    # True once the encoder has been (or starts as) unfrozen
    encoder_unfrozen = freeze_enc_mode == "no"
    history = {
        "diff_loss": [],
        "seg_sup": [],
        "seg_unsup": [],
        "seg_view": [],
        "val_dice": [],
        "val_iou": [],
        # raw (unweighted) sup-loss components
        "sup_bce": [],
        "sup_dice": [],
        "sup_cldice": [],
        "sup_bound": [],
        # raw (unweighted) unsup-loss components.
        # total loss_unsup (raw) = loss_refined_unsup + loss_view_unsup.
        # Weighted contribution to the objective is:
        #   pseudo_w * unsup_refined + lambda_view * unsup_view
        # so unsup_refined, unsup_view and pseudo_weight are enough to
        # reconstruct the epoch's weighted loss_unsup (lambda_view from yaml).
        "unsup_refined": [],
        "unsup_view": [],
        "unsup_total": [],
        "unsup_rect": [],   # diffrect only (weak_weak term)
        "unsup_fix": [],    # diffrect only (strong_weak term)
        "pseudo_weight": [],
    }
    grad_clip = tr.get("grad_clip", 1.0)
    noise_frac = tr["noise_frac"]
    # Confidence-threshold curriculum (hybrid: strict warmup -> percentile)
    tau_max = float(loss_c.get("tau_max", 0.95))
    tau_min = float(loss_c.get("tau_min", 0.70))
    conf_strict_epochs = int(loss_c.get("conf_strict_epochs", 5))
    keep_frac_min = float(loss_c.get("keep_frac_min", 0.10))
    keep_frac_max = float(loss_c.get("keep_frac_max", 0.60))

    for epoch in range(1, epochs + 1):
        is_warmup_phase = (epoch <= warmup_epochs) and is_warmup
        strict, keep_frac = get_conf_schedule(
            epoch, epochs, warmup_epochs, is_warmup,
            conf_strict_epochs, keep_frac_min, keep_frac_max,
        )
        phase = "WARMUP" if is_warmup_phase else "SEMI-SUP"
        conf_mode = f"strict(tau_max={tau_max:.2f})" if strict else f"percentile(keep={keep_frac:.2f})"
        logger.info("Epoch %d/%d [%s] conf=%s", epoch, epochs, phase, conf_mode)

        # Unfreeze encoder when transitioning out of warmup (freeze_enc="partly")
        if freeze_enc_mode == "partly" and not is_warmup_phase and not encoder_unfrozen:
            seg_model.unfreeze_encoder()
            seg_opt = torch.optim.AdamW(
                seg_model.parameters(), 
                lr=seg_c["lr_unfrozen"], 
                weight_decay=seg_c["weight_decay_unfrozen"],
            )
            seg_lrsched = torch.optim.lr_scheduler.CosineAnnealingLR(
                seg_opt, epochs - epoch + 1, eta_min=1e-7
            )
            encoder_unfrozen = True
            no_improve = 0
            logger.info("Encoder unfrozen at epoch %d — patience counter reset", epoch)

        # On the first epoch of semi-sup, reset the LR schedule and patience counter
        # so the full cosine-annealing budget is available for the unsupervised phase.
        if is_warmup and epoch == warmup_epochs + 1:
            seg_lrsched = torch.optim.lr_scheduler.CosineAnnealingLR(
                seg_opt, epochs - warmup_epochs, eta_min=1e-7
            )
            no_improve = 0
            logger.info(
                "Entering semi-sup at epoch %d — LR scheduler reset, patience counter reset",
                epoch,
            )

        if use_diffusion:
            # --- Diffusion training pass ---
            diff_model.train()
            total_diff_train = 0.0
            for batch in tqdm(diff_ld, desc="Diff-Train", leave=False):
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                t = torch.randint(0, diff_sched.T, (img.shape[0],), device=device)
                xt, eps = diff_sched.q_sample(mask, t)
                loss = F.mse_loss(diff_model(xt, t, img), eps)
                diff_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diff_model.parameters(), grad_clip)
                diff_opt.step()
                total_diff_train += loss.item()

            diff_lrsched.step()
            avg_diff_train = total_diff_train / max(len(diff_ld), 1)

            # --- Diffusion validation pass (held-out val images) ---
            # Saves the checkpoint that generalises best, not the one that
            # memorises the (few) training samples most.
            diff_model.eval()
            total_diff_val = 0.0
            with torch.no_grad():
                for batch in diff_val_ld:
                    img = batch["image"].to(device)
                    mask = batch["mask"].to(device)
                    t = torch.randint(0, diff_sched.T, (img.shape[0],), device=device)
                    xt, eps = diff_sched.q_sample(mask, t)
                    total_diff_val += F.mse_loss(diff_model(xt, t, img), eps).item()
            diff_model.train()
            avg_diff_val = total_diff_val / max(len(diff_val_ld), 1)
            avg_diff = avg_diff_val  # use val loss as the epoch-level metric

            logger.info(
                "Diffusion: train_MSE=%.4f  val_MSE=%.4f  (%d train / %d val batches)",
                avg_diff_train, avg_diff_val, len(diff_ld), len(diff_val_ld),
            )
            history["diff_loss"].append(avg_diff)
            if avg_diff_val < best_diff_loss:
                best_diff_loss = avg_diff_val
                torch.save(
                    {"model": diff_model.state_dict(), "epoch": epoch, "loss": avg_diff_val},
                    f"{ckpt_dir}/diffusion_best.pt",
                )
                logger.info("Saved diffusion_best.pt (new best val_MSE=%.4f)", avg_diff_val)
        else:
            avg_diff = 0.0
            history["diff_loss"].append(avg_diff)

        # Epoch-level constants — compute once, reuse across all mini-batches
        pseudo_w = get_pseudo_weight(
            epoch, epochs, loss_c["lambda_pseudo"],
            warmup_epochs if is_warmup else 0,
        )
        lambda_view = loss_c["lambda_view"]

        seg_model.train()
        total_sup = total_refined_unsup = total_view = 0.0
        total_bce = total_dice = total_cldice = total_bound = 0.0
        total_view_unsup = 0.0
        total_rect = total_fix = 0.0
        skipped = 0
        valid_px_ratios = []
        conf_thresh_vals = []
        unsup_iter = iter(unsup_ld)

        seg_opt.zero_grad(set_to_none=True)
        accum_count = 0

        for sup_batch in tqdm(sup_ld, desc="Segmentation", leave=False):
            img_a = sup_batch["image_a"].to(device)
            img_b = sup_batch["image_b"].to(device)
            mask = sup_batch["mask"].to(device)
            txt = get_text_emb(img_a.shape[0])

            amp_ctx = autocast() if use_amp else nullcontext()
            with amp_ctx:
                logits_a, feats_a = seg_model(img_a, txt)
                l_sup, sup_comps = seg_loss_fn(logits_a, mask)
                _, feats_b = seg_model(img_b, txt)
                l_view_sup = None

                loss_refined_unsup = None
                loss_view_unsup = None
                if not is_warmup_phase:
                    try:
                        unsup_batch = next(unsup_iter)
                    except StopIteration:
                        unsup_iter = iter(unsup_ld)
                        unsup_batch = next(unsup_iter)

                    img_ua = unsup_batch["image_a"].to(device)
                    img_ub = unsup_batch["image_b"].to(device)
                    txt_u = get_text_emb(img_ua.shape[0])

                    if use_diffusion:
                        output = (
                            compute_diffusion_unsup_losses(
                                seg_model,
                                diff_model,
                                diff_sched,
                                img_ua,
                                img_ub,
                                txt_u,
                                device,
                                get_text_emb,
                                noise_frac,
                                strict,
                                keep_frac,
                                tau_min,
                                tau_max,
                                pseudo_ablation,
                                consistency_type,
                                lambda_soft,
                                ema_model,   # teacher: EMA model if enabled, else None → falls back to seg_model
                            )
                        )
                        if pseudo_ablation == "diffrect":
                            loss_refined_unsup, (l_rect, l_fix), loss_view_unsup, vp_ratio, conf_eff, batch_skipped = output
                        else:
                            loss_refined_unsup, loss_view_unsup, vp_ratio, conf_eff, batch_skipped = output

                        valid_px_ratios.append(vp_ratio)
                        conf_thresh_vals.append(conf_eff)
                        if batch_skipped:
                            skipped += 1
                    else:
                        # No diffusion: unlabeled branch uses ONLY cross-view
                        # consistency (loss_view_unsup). No refined pseudo-label
                        # term — loss_refined_unsup stays None.
                        logits_ua, _ = seg_model(img_ua, txt_u)
                        logits_ub, _ = seg_model(img_ub, txt_u)
                        loss_view_unsup = view_consistency_combined(
                            logits_ua, logits_ub, consistency_type, lambda_soft
                        )
                        loss_refined_unsup = None

                if is_warmup_phase:
                    # Warmup: supervised loss + view-consistency on labeled pairs
                    total_loss = l_sup
                elif use_diffusion:
                    # Semi-sup with diffusion:
                    #   total loss_unsup = pseudo_w * loss_refined_unsup
                    #                      + lambda_view * loss_view_unsup
                    total_loss = l_sup
                    if loss_refined_unsup is not None:
                        total_loss = total_loss + pseudo_w * loss_refined_unsup
                    if loss_view_unsup is not None:
                        total_loss = total_loss + lambda_view * loss_view_unsup
                else:
                    # Semi-sup without diffusion: sup + view-unsup only
                    total_loss = l_sup
                    if loss_view_unsup is not None:
                        total_loss = total_loss + lambda_view * loss_view_unsup

            if not torch.isfinite(total_loss):
                continue

            loss_part = total_loss / accum_steps
            if use_amp and seg_scaler is not None:
                seg_scaler.scale(loss_part).backward()
            else:
                loss_part.backward()

            accum_count += 1
            total_sup   += l_sup.item()
            if l_view_sup is not None:
                total_view += l_view_sup.item()
            total_bce   += sup_comps[0]
            total_dice  += sup_comps[1]
            total_cldice += sup_comps[2]
            total_bound += sup_comps[3]
            if loss_refined_unsup is not None:
                total_refined_unsup += loss_refined_unsup.item()
            if loss_view_unsup is not None:
                total_view_unsup += loss_view_unsup.item()
            if not is_warmup_phase and use_diffusion and pseudo_ablation == "diffrect":
                if l_rect is not None:
                    total_rect += l_rect.item()
                if l_fix is not None:
                    total_fix += l_fix.item()

            if accum_count >= accum_steps:
                if use_amp and seg_scaler is not None:
                    seg_scaler.unscale_(seg_opt)
                    torch.nn.utils.clip_grad_norm_(seg_model.parameters(), grad_clip)
                    seg_scaler.step(seg_opt)
                    seg_scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(seg_model.parameters(), grad_clip)
                    seg_opt.step()
                if ema_model is not None:
                    _ema_update(ema_model, seg_model, ema_alpha)
                seg_opt.zero_grad(set_to_none=True)
                accum_count = 0

        if accum_count > 0:
            if use_amp and seg_scaler is not None:
                seg_scaler.unscale_(seg_opt)
                torch.nn.utils.clip_grad_norm_(seg_model.parameters(), grad_clip)
                seg_scaler.step(seg_opt)
                seg_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(seg_model.parameters(), grad_clip)
                seg_opt.step()
            if ema_model is not None:
                _ema_update(ema_model, seg_model, ema_alpha)
            seg_opt.zero_grad(set_to_none=True)

        seg_lrsched.step()
        n_sup = max(len(sup_ld), 1)
        history["seg_sup"].append(total_sup / n_sup)
        history["seg_view"].append(total_view / n_sup)
        history["sup_bce"].append(total_bce / n_sup)
        history["sup_dice"].append(total_dice / n_sup)
        history["sup_cldice"].append(total_cldice / n_sup)
        history["sup_bound"].append(total_bound / n_sup)
        if not is_warmup_phase:
            denom_unsup = max(n_sup - skipped, 1)
            refined_avg = total_refined_unsup / denom_unsup
            view_avg = total_view_unsup / n_sup
            history["unsup_refined"].append(refined_avg)
            history["unsup_view"].append(view_avg)
            history["unsup_total"].append(refined_avg + view_avg)
            history["unsup_rect"].append(total_rect / denom_unsup)
            history["unsup_fix"].append(total_fix / denom_unsup)
            # seg_unsup is the FULL unsup loss actually added to the objective,
            # i.e. the weighted sum of both terms (not the refined term alone):
            #   seg_unsup = pseudo_w * loss_refined_unsup + lambda_view * loss_view_unsup
            history["seg_unsup"].append(pseudo_w * refined_avg + lambda_view * view_avg)
        else:
            for _k in ("unsup_refined", "unsup_view", "unsup_total",
                       "unsup_rect", "unsup_fix"):
                history[_k].append(0.0)
            history["seg_unsup"].append(0.0)
        history["pseudo_weight"].append(pseudo_w)
        avg_vp = float(np.mean(valid_px_ratios)) if valid_px_ratios else 0.0
        avg_conf = float(np.mean(conf_thresh_vals)) if conf_thresh_vals else 0.0

        metrics = evaluate(seg_model, val_ld, device, get_text_emb)
        history["val_dice"].append(metrics["dice"])
        history["val_iou"].append(metrics["iou"])

        logger.info(
            "diff=%.4f sup=%.4f unsup_w=%.4f val_dice=%.4f iou=%.4f valid_px=%.3f conf_thr=%.3f skipped=%d",
            avg_diff,
            history["seg_sup"][-1],
            history["seg_unsup"][-1],
            metrics["dice"],
            metrics["iou"],
            avg_vp,
            avg_conf,
            skipped,
        )
        logger.info(
            "  sup components (raw) → bce=%.4f  dice=%.4f  cldice=%.4f  bound=%.4f",
            history["sup_bce"][-1],
            history["sup_dice"][-1],
            history["sup_cldice"][-1],
            history["sup_bound"][-1],
        )
        logger.info(
            "  unsup components (raw) → refined=%.4f  view=%.4f  total=%.4f  (pseudo_w=%.4f, lambda_view=%.4f)",
            history["unsup_refined"][-1],
            history["unsup_view"][-1],
            history["unsup_total"][-1],
            pseudo_w,
            lambda_view,
        )
        if pseudo_ablation == "diffrect":
            logger.info(
                "  unsup refined split (raw) → rect=%.4f  fix=%.4f",
                history["unsup_rect"][-1],
                history["unsup_fix"][-1],
            )

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            no_improve = 0
            torch.save(seg_model.state_dict(), f"{ckpt_dir}/seg_{cfg['dataset']}_uni_conch_best.pt")
            logger.info("New best val IoU: %.4f", best_iou)
        else:
            no_improve += 1
            logger.info("No improvement (%d/%d)", no_improve, tr["patience"])
            if no_improve >= tr["patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ----------------------------- Training curves --------------------------
    # Number of warmup epochs actually used (0 when warmup disabled). During
    # warmup the unsup loss is 0; we blank the single transition point so the
    # curve doesn't draw a steep diagonal from 0 up to the first real value.
    n_warmup = warmup_epochs if is_warmup else 0
    lb = loss_c.get("lambda_bce", 1.0)
    ld = loss_c.get("lambda_dice", 1.0)
    lc = loss_c["lambda_cldice"]
    lbo = loss_c["lambda_bound"]
    lv = loss_c["lambda_view"]

    def _arr(key: str) -> np.ndarray:
        return np.array(history[key], dtype=float)

    def _break_warmup(values: np.ndarray) -> np.ndarray:
        arr = np.array(values, dtype=float)
        if 0 < n_warmup < len(arr):
            arr[n_warmup - 1] = np.nan  # blank warmup→semi-sup transition
        return arr

    def _mark_max(ax, values, color: str) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return
        idx = int(np.nanargmax(arr))
        ax.scatter([idx], [arr[idx]], color=color, s=140, marker="*",
                   zorder=5, edgecolors="black", linewidths=0.6)
        ax.annotate(f"{arr[idx]:.4f} @ ep{idx + 1}", (idx, arr[idx]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=color)

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 2)
    ax_diff = fig.add_subplot(gs[0, 0])
    ax_seg = fig.add_subplot(gs[0, 1])
    ax_sup = fig.add_subplot(gs[1, :])        # spans [1,0] and [1,1]
    ax_unsup = fig.add_subplot(gs[2, 0])
    ax_val = fig.add_subplot(gs[2, 1])

    # [0,0] diffusion loss
    ax_diff.plot(history["diff_loss"], "b-o")
    ax_diff.set_title("Diffusion Loss")
    ax_diff.set_xlabel("epoch")
    ax_diff.grid(True)

    # [0,1] weighted seg losses; total = the loss used for the gradient step
    sup_w = _arr("seg_sup")
    unsup_w = _arr("seg_unsup")
    total_w = sup_w + unsup_w
    ax_seg.plot(sup_w, "g-o", label="loss_sup")
    ax_seg.plot(_break_warmup(unsup_w), "r-s", label="loss_unsup")
    ax_seg.plot(total_w, "k-^", label="loss_total")
    ax_seg.legend()
    ax_seg.set_title("Seg Losses")
    ax_seg.set_xlabel("epoch")
    ax_seg.grid(True)

    # [1,:] weighted sup components — these sum to loss_sup
    ax_sup.plot(_arr("sup_bce") * lb,    "b-o", label=f"BCE")
    ax_sup.plot(_arr("sup_dice") * ld,   "g-s", label=f"Dice")
    ax_sup.plot(_arr("sup_cldice") * lc, "r-^", label=f"CLDice")
    ax_sup.plot(_arr("sup_bound") * lbo, "m-d", label=f"Boundary")
    ax_sup.legend(ncol=4)
    ax_sup.set_title("Sup Components")
    ax_sup.set_xlabel("epoch")
    ax_sup.grid(True)

    # [2,0] weighted unsup components (refined x pseudo_w, view x lambda_view)
    refined_w = _arr("unsup_refined") * _arr("pseudo_weight")
    view_w = _arr("unsup_view") * lv
    ax_unsup.plot(_break_warmup(refined_w), "c-o", label="loss_refined")
    ax_unsup.plot(_break_warmup(view_w), "y-s", label=f"loss_consistency_view")
    ax_unsup.legend()
    ax_unsup.set_title("Unsup Components")
    ax_unsup.set_xlabel("epoch")
    ax_unsup.grid(True)

    # [2,1] validation Dice & IoU, best value highlighted on each line
    ax_val.plot(history["val_dice"], color="purple", marker="o", label="val Dice")
    ax_val.plot(history["val_iou"], color="orange", marker="s", label="val IoU")
    _mark_max(ax_val, history["val_dice"], "purple")
    _mark_max(ax_val, history["val_iou"], "orange")
    ax_val.legend()
    ax_val.set_title("Val Dice & IoU")
    ax_val.set_xlabel("epoch")
    ax_val.grid(True)

    plt.tight_layout()
    plot_path = Path(log_dir) / "training_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves to %s", plot_path)

    logger.info("Training done. Best val IoU: %.4f", best_iou)
    return diff_model, seg_model
