import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from datasets.dataloader import IMG_EXTS, resolve_split_dirs
from evaluation.metrics import compute_dice_np, compute_iou_np
from models.backbone import build_conch_text_encoder, build_segmentation_model
from scripts.inference import infer_full_image

logger = logging.getLogger(__name__)


def _list_images(img_dir: str) -> List[str]:
    paths = []
    for ext in sorted(IMG_EXTS):
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(paths, key=lambda p: Path(p).name)


def _mask_path_for_stem(mask_dir: Optional[str], stem: str) -> Optional[str]:
    if not mask_dir or not os.path.isdir(mask_dir):
        return None
    for ext in sorted(IMG_EXTS):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def run_test_evaluation(
    cfg: Dict[str, Any],
    device: torch.device,
    weights: Path,
    project_root: Path,
    log: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Full-resolution test: sliding-window inference on paths.data_root / test.
    Works for normalized (.../test/images) and raw (.../test/image).
    """
    log = log or logger
    data_root = cfg["paths"]["data_root"]
    resolved = resolve_split_dirs(data_root)
    if "test" not in resolved:
        log.warning("No test split under %s; skip test evaluation.", data_root)
        return {}
    test_img_dir, test_mask_dir = resolved["test"]
    if not os.path.isdir(test_img_dir):
        log.warning("Test image dir missing: %s", test_img_dir)
        return {}

    inf = cfg.get("infer", {})
    ps, st = inf.get("patch_size", 256), inf.get("stride", 128)

    text_cfg = cfg.get("text") or {}
    use_text_encoder = bool(text_cfg.get("use_text_encoder", True))

    if use_text_encoder:
        text_enc = build_conch_text_encoder(cfg, device)
        def get_text_emb(b: int):
            return text_enc(b)
    else:
        def get_text_emb(b: int):
            return None

    model = build_segmentation_model(cfg, freeze_enc=False).to(device)
    wpath = weights if weights.is_absolute() else project_root / weights
    if not wpath.is_file():
        log.warning("Checkpoint not found: %s; skip test evaluation.", wpath)
        return {}

    raw_sd = torch.load(wpath, map_location=device)
    model_sd = model.state_dict()

    # Filter: keep only keys present in the current model with matching shapes.
    # This makes loading robust to architecture changes between training runs.
    compatible, skipped_missing, skipped_shape = {}, [], []
    for k, v in raw_sd.items():
        if k not in model_sd:
            skipped_missing.append(k)
        elif v.shape != model_sd[k].shape:
            skipped_shape.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}")
        else:
            compatible[k] = v

    if skipped_missing:
        log.warning(
            "Checkpoint has %d key(s) not in current model (ignored): %s%s",
            len(skipped_missing),
            ", ".join(skipped_missing[:5]),
            " ..." if len(skipped_missing) > 5 else "",
        )
    if skipped_shape:
        log.warning(
            "Checkpoint has %d key(s) with shape mismatch (ignored, random init used): %s%s",
            len(skipped_shape),
            "; ".join(skipped_shape[:3]),
            " ..." if len(skipped_shape) > 3 else "",
        )

    missing_in_ckpt = [k for k in model_sd if k not in raw_sd]
    if missing_in_ckpt:
        log.warning(
            "%d model key(s) not in checkpoint (random init): %s%s",
            len(missing_in_ckpt),
            ", ".join(missing_in_ckpt[:5]),
            " ..." if len(missing_in_ckpt) > 5 else "",
        )

    model.load_state_dict(compatible, strict=False)
    log.info(
        "Loaded %d/%d compatible weight tensors from %s",
        len(compatible), len(model_sd), wpath.name,
    )
    model.eval()

    test_imgs = _list_images(test_img_dir)
    results = []
    for img_path in tqdm(test_imgs, desc="test"):
        stem = Path(img_path).stem
        mask_path = _mask_path_for_stem(test_mask_dir, stem)
        _, gt, pred = infer_full_image(
            model, img_path, mask_path, device, get_text_emb, patch_size=ps, stride=st
        )
        if gt is not None:
            results.append(
                {
                    "dice": compute_dice_np(pred, gt),
                    "iou": compute_iou_np(pred, gt),
                }
            )

    if not results:
        log.warning("No test pairs scored (missing masks?).")
        return {}

    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]
    summary = {
        "test_dice_mean": float(np.mean(dices)),
        "test_dice_std": float(np.std(dices)),
        "test_iou_mean": float(np.mean(ious)),
        "test_iou_std": float(np.std(ious)),
        "test_n": float(len(results)),
    }
    log.info(
        "Test (%d images) Dice: %.4f ± %.4f | IoU: %.4f ± %.4f",
        len(results),
        summary["test_dice_mean"],
        summary["test_dice_std"],
        summary["test_iou_mean"],
        summary["test_iou_std"],
    )
    return summary
