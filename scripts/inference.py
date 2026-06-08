import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.transforms import MEAN, STD
from models.backbone import build_conch_text_encoder, build_segmentation_model
from utils.checkpoint import load_yaml_config, resolve_project_paths
from utils.hf_auth import configure_hf_token
from utils.logger import setup_logging
from utils.seed import set_seed


@torch.no_grad()
def infer_full_image(
    model: torch.nn.Module,
    img_path: str,
    mask_path: Optional[str],
    device: torch.device,
    get_text_emb: Callable[[int], torch.Tensor],
    patch_size: int = 256,
    stride: int = 128,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    norm_tf = A.Compose([A.Normalize(MEAN, STD), ToTensorV2()])
    pred_map = np.zeros((h, w), dtype=np.float32)
    wt_map = np.zeros((h, w), dtype=np.float32)
    gauss = gaussian_filter(np.ones((patch_size, patch_size), dtype=np.float32), sigma=patch_size // 8)
    ph = max(0, (patch_size - h % patch_size) % patch_size)
    pw = max(0, (patch_size - w % patch_size) % patch_size)
    img_pad = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    r = 0
    while r + patch_size <= img_pad.shape[0]:
        c = 0
        while c + patch_size <= img_pad.shape[1]:
            patch = img_pad[r : r + patch_size, c : c + patch_size]
            t = norm_tf(image=patch)["image"].unsqueeze(0).to(device)
            txt = get_text_emb(1)
            logit, _ = model(t, txt)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            re = min(r + patch_size, h)
            ce = min(c + patch_size, w)
            pred_map[r:re, c:ce] += prob[: re - r, : ce - c] * gauss[: re - r, : ce - c]
            wt_map[r:re, c:ce] += gauss[: re - r, : ce - c]
            c += stride
        r += stride
    pred_map = pred_map / (wt_map + 1e-6)
    gt = (
        (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        if mask_path
        else None
    )
    return img_rgb, gt, pred_map


def main():
    parser = argparse.ArgumentParser(description="Run sliding-window inference on one image")
    parser.add_argument("--config", type=str, default="configs/crag.yaml")
    parser.add_argument("--weights", type=str, required=True, help="seg_crag_uni_conch_best.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, default="", help="optional GT mask for visualization")
    parser.add_argument("--out", type=str, default="pred.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()
    project_root = ROOT
    cfg = load_yaml_config(str(project_root / args.config))
    cfg = resolve_project_paths(cfg, project_root)
    configure_hf_token(cfg)
    set_seed(cfg.get("seed", 42))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text_enc = build_conch_text_encoder(cfg, device)

    def get_text_emb(b: int) -> torch.Tensor:
        return text_enc(b)

    model = build_segmentation_model(cfg, freeze_enc=False).to(device)
    wpath = Path(args.weights)
    if not wpath.is_absolute():
        wpath = project_root / wpath
    model.load_state_dict(torch.load(wpath, map_location=device), strict=True)
    model.eval()

    inf = cfg.get("infer", {})
    ps = inf.get("patch_size", 256)
    st = inf.get("stride", 128)

    mask_path = args.mask or None
    _, gt, pred = infer_full_image(
        model, args.image, mask_path, device, get_text_emb, patch_size=ps, stride=st
    )
    cv2.imwrite(args.out, (pred > 0.5).astype(np.uint8) * 255)
    print(f"Saved binary mask to {args.out}")
    if gt is not None:
        from evaluation.metrics import compute_dice_np, compute_iou_np

        d = compute_dice_np(pred, gt)
        i = compute_iou_np(pred, gt)
        print(f"Dice={d:.4f} IoU={i:.4f}")


if __name__ == "__main__":
    main()
