import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from PIL import Image

# Ensure local package imports work when running this file directly
import sys

HERE = os.path.dirname(__file__)
PKG_ROOT = os.path.dirname(HERE)
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

from datasets.dataloader import find_mask_for_image
from datasets.online_aug import MEAN, STD
from models import (
    build_conch_text_encoder,
    build_segmentation_model,
    DiffusionScheduler,
    MiniUNet,
)

# ---------------------------------------------------------------------------
# Đặt đường dẫn ảnh / mask trực tiếp ở đây (không truyền --image / --mask trên CLI).
# MASK_PATH = None → tự tìm mask cạnh ảnh (vd. test/image/x.tif → test/mask/x.png).
# ---------------------------------------------------------------------------
IMAGE_PATH = "/home/thaivan/Desktop/research-2026/thaivan/SFC/dataset/crag/10/test/image/test_2.tif"
MASK_PATH: Optional[str] = "/home/thaivan/Desktop/research-2026/thaivan/SFC/dataset/crag/10/test/mask/test_2.png"

# Sliding window: stride None → patch_size // 2 (có overlap, trung bình vùng gối).
VIZ_SLIDE_STRIDE: Optional[int] = None
# Số tile seg chạy một lần (tăng nếu VRAM đủ).
VIZ_TILE_BATCH_SEG = 4


def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is not None and bgr.ndim == 3:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception:
        raise FileNotFoundError(f"Cannot read image (unsupported or corrupted): {os.path.abspath(path)}")


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def auto_resolve_gt_mask_path(image_path: str) -> Optional[str]:
    ip = Path(image_path).resolve()
    if ip.parent.name.lower() in ("image", "images"):
        for sub in ("mask", "masks"):
            md = ip.parent.parent / sub
            if md.is_dir():
                hit = find_mask_for_image(str(md), str(ip))
                if hit and os.path.isfile(hit):
                    return os.path.abspath(hit)
    hit = find_mask_for_image(str(ip.parent), str(ip))
    if hit and os.path.isfile(hit):
        return os.path.abspath(hit)
    return None


def read_mask_gray(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if m.shape[0] != shape_hw[0] or m.shape[1] != shape_hw[1]:
        raise ValueError(
            f"Mask shape {m.shape} != image shape {shape_hw}; resize mask or use matching pair."
        )
    return m


def pad_min_patch(img: np.ndarray, mask: Optional[np.ndarray], patch: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pad bottom/right (reflect img, 0 mask) đủ để H,W >= patch."""
    H, W = img.shape[:2]
    ph = max(0, patch - H)
    pw = max(0, patch - W)
    if ph == 0 and pw == 0:
        return img, mask
    img_p = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    mask_p = None
    if mask is not None:
        mask_p = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    return img_p, mask_p


def crop_to_original(x: np.ndarray, h0: int, w0: int) -> np.ndarray:
    return x[:h0, :w0].copy()


def tile_starts(length: int, patch: int, stride: int) -> List[int]:
    if length <= patch:
        return [0]
    starts: List[int] = []
    pos = 0
    while pos + patch <= length:
        starts.append(pos)
        pos += stride
    last = length - patch
    if not starts:
        return [0]
    if starts[-1] < last:
        starts.append(last)
    return starts


def patch_to_imagenet(patch_hwc: np.ndarray) -> torch.Tensor:
    x = patch_hwc.astype(np.float32) / 255.0
    for c in range(3):
        x[:, :, c] = (x[:, :, c] - MEAN[c]) / STD[c]
    return torch.from_numpy(x).permute(2, 0, 1)


def iter_tile_coords(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    return [(y, x) for y in tile_starts(h, patch, stride) for x in tile_starts(w, patch, stride)]


@torch.no_grad()
def sliding_seg_probability(
    seg_model: torch.nn.Module,
    img_p: np.ndarray,
    patch: int,
    stride: int,
    device: torch.device,
    txt_1: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Trả về map xác suất [H,W] (ảnh đã pad)."""
    hp, wp = img_p.shape[:2]
    coords = iter_tile_coords(hp, wp, patch, stride)
    acc = np.zeros((hp, wp), dtype=np.float64)
    wts = np.zeros((hp, wp), dtype=np.float64)
    seg_model.eval()
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        tiles = torch.stack(
            [patch_to_imagenet(img_p[y : y + patch, x : x + patch]) for y, x in batch_coords],
            dim=0,
        ).to(device)
        b = tiles.shape[0]
        te = txt_1.expand(b, -1, -1)
        logits, _ = seg_model(tiles, te)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        for k, (y, x) in enumerate(batch_coords):
            acc[y : y + patch, x : x + patch] += prob[k]
            wts[y : y + patch, x : x + patch] += 1.0
    out = (acc / np.maximum(wts, 1e-6)).astype(np.float32)
    return out


@torch.no_grad()
def sliding_diffusion_refine(
    diff_model: torch.nn.Module,
    diff_sched: DiffusionScheduler,
    pseudo_p: np.ndarray,
    img_p: np.ndarray,
    patch: int,
    stride: int,
    t_start: int,
    device: torch.device,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mỗi tile: x0 từ pseudo, add noise đến t_start, denoise về 0; ghép (trung bình overlap).

    Nhiễu forward và nhiễu stochastic khi denoise lấy từ **một** Gaussian toàn cục (hp×wp)
    rồi cắt theo tile — tránh mỗi tile rand riêng gây lưỡi cưa/grid trên ảnh noisy.
    """
    hp, wp = img_p.shape[:2]
    coords = iter_tile_coords(hp, wp, patch, stride)
    acc_n = np.zeros((hp, wp), dtype=np.float64)
    acc_r = np.zeros((hp, wp), dtype=np.float64)
    wts = np.zeros((hp, wp), dtype=np.float64)
    diff_model.eval()

    set_all_seeds(base_seed)
    noise_q = torch.randn(1, 1, hp, wp, device=device, dtype=torch.float32)
    # Một map nhiễu cho mỗi bước DDPM có stochastic term (t > 0).
    z_steps: List[torch.Tensor] = [
        torch.randn(1, 1, hp, wp, device=device, dtype=torch.float32)
        for _ in range(max(0, t_start - 1))
    ]

    for y, x in coords:
        tile_rgb = img_p[y : y + patch, x : x + patch]
        tile_p = pseudo_p[y : y + patch, x : x + patch]
        img_b = patch_to_imagenet(tile_rgb).unsqueeze(0).to(device)
        x0 = torch.from_numpy(tile_p).float().view(1, 1, patch, patch).to(device) * 2.0 - 1.0
        noise = noise_q[:, :, y : y + patch, x : x + patch]
        ab_t = diff_sched.alphas_bar[t_start]
        xt = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * noise
        noisy01 = torch.clamp((xt + 1) / 2, 0, 1)[0, 0].cpu().numpy()
        cur = xt.clone()
        zi = 0
        for t in reversed(range(t_start)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            eps_pred = diff_model(cur, t_batch, img_b)
            ab_tt = diff_sched.alphas_bar[t]
            alpha_t = ab_tt / diff_sched.alphas_bar[t - 1] if t > 0 else ab_tt
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - ab_tt)
            mean = coef1 * (cur - coef2 * eps_pred)
            if t > 0:
                ab_prev = diff_sched.alphas_bar[t - 1]
                sigma = torch.sqrt((1 - ab_prev) / (1 - ab_tt) * (1 - alpha_t))
                z = z_steps[zi][:, :, y : y + patch, x : x + patch]
                zi += 1
                cur = mean + sigma * z
            else:
                cur = mean
        refined01 = torch.clamp((cur + 1) / 2, 0, 1)[0, 0].cpu().numpy()
        acc_n[y : y + patch, x : x + patch] += noisy01
        acc_r[y : y + patch, x : x + patch] += refined01
        wts[y : y + patch, x : x + patch] += 1.0
    w = np.maximum(wts, 1e-6)
    return (acc_n / w).astype(np.float32), (acc_r / w).astype(np.float32)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Visualize training pipeline on a single image (full-res sliding window)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (crag.yaml/glas.yaml)")
    parser.add_argument("--seg-weights", type=str, required=True, help="Path to segmentation checkpoint .pt")
    parser.add_argument("--diff-weights", type=str, required=True, help="Path to diffusion checkpoint .pt")
    parser.add_argument("--out-dir", type=str, default="./viz_out", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for global diffusion noise fields (shared across tiles, avoids tile seams).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    ensure_dir(args.out_dir)

    if not (IMAGE_PATH and str(IMAGE_PATH).strip()):
        raise RuntimeError(
            "Set IMAGE_PATH at the top of scripts/visualize_pipeline.py to your RGB image path."
        )
    image_path = resolve_image_path(IMAGE_PATH)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"IMAGE_PATH not found: {image_path}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    patch_size = int(cfg.get("patch", {}).get("size", 256))
    stride = VIZ_SLIDE_STRIDE if VIZ_SLIDE_STRIDE is not None else max(1, patch_size // 2)
    noise_frac = float(cfg.get("train", {}).get("noise_frac", 0.4))
    diff_c = cfg.get("diffusion", {}) or {}
    T = int(diff_c.get("T", 100))
    t_start = int(T * noise_frac)

    img_rgb = imread_rgb(image_path)
    h0, w0 = img_rgb.shape[:2]

    mask_path_resolved: Optional[str] = None
    mask_gray: Optional[np.ndarray] = None
    if MASK_PATH and str(MASK_PATH).strip():
        mp = os.path.abspath(os.path.expanduser(MASK_PATH))
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"MASK_PATH not found: {mp}")
        mask_path_resolved = mp
        mask_gray = read_mask_gray(mp, (h0, w0))
    else:
        mask_path_resolved = auto_resolve_gt_mask_path(image_path)
        if mask_path_resolved:
            try:
                mask_gray = read_mask_gray(mask_path_resolved, (h0, w0))
            except ValueError as e:
                print(f"Warning: skip auto mask (shape mismatch): {e}")
                mask_path_resolved = None
                mask_gray = None

    img_p, mask_p = pad_min_patch(img_rgb, mask_gray, patch_size)

    text_enc = build_conch_text_encoder(cfg, device=device)
    seg_model = build_segmentation_model(cfg, freeze_enc=True, grad_checkpointing=False).to(device)
    seg_model.eval()
    diff_model = MiniUNet(
        base_ch=int(diff_c.get("base_ch", 64)),
        depth=int(diff_c.get("depth", 4)),
        T=T,
    ).to(device)
    diff_model.eval()
    diff_sched = DiffusionScheduler(T=T, device=device)

    seg_state = torch.load(args.seg_weights, map_location=device)
    if isinstance(seg_state, dict) and "model" in seg_state:
        seg_model.load_state_dict(seg_state["model"], strict=False)
    else:
        seg_model.load_state_dict(seg_state, strict=False)
    diff_state = torch.load(args.diff_weights, map_location=device)
    if isinstance(diff_state, dict) and "model" in diff_state:
        diff_model.load_state_dict(diff_state["model"], strict=False)
    else:
        diff_model.load_state_dict(diff_state, strict=False)

    txt_1 = text_enc.forward(batch_size=1)

    set_all_seeds(args.seed)
    pseudo_p = sliding_seg_probability(
        seg_model, img_p, patch_size, stride, device, txt_1, VIZ_TILE_BATCH_SEG
    )

    noisy_p, refined_p = sliding_diffusion_refine(
        diff_model,
        diff_sched,
        pseudo_p,
        img_p,
        patch_size,
        stride,
        t_start,
        device,
        args.seed,
    )

    # Cắt về kích thước gốc (bỏ vùng pad nếu có)
    pseudo_full = crop_to_original(pseudo_p, h0, w0)
    noisy_full = crop_to_original(noisy_p, h0, w0)
    refined_full = crop_to_original(refined_p, h0, w0)

    out = args.out_dir
    cv2.imwrite(os.path.join(out, "input_full.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    if mask_gray is not None:
        gt_scaled = mask_gray.astype(np.float32) / 127.5 - 1.0
        gt_m11_01 = np.clip((gt_scaled + 1.0) / 2.0, 0.0, 1.0)
        cv2.imwrite(os.path.join(out, "gt_mask_scaled_minus1_to1_vis.png"), to_uint8(gt_m11_01))

    pseudo_u8 = to_uint8(pseudo_full)
    noisy_u8 = to_uint8(noisy_full)
    refined_u8 = to_uint8(refined_full)

    cv2.imwrite(os.path.join(out, "pseudo_before_refine_full.png"), pseudo_u8)
    cv2.imwrite(os.path.join(out, "pseudo_mask_after_noise_full.png"), noisy_u8)
    cv2.imwrite(os.path.join(out, "pseudo_after_refine_full.png"), refined_u8)
    cv2.imwrite(os.path.join(out, "diffusion_cond_full.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Tên cũ (cùng nội dung full-res)
    cv2.imwrite(os.path.join(out, "pseudo_before_refine.png"), pseudo_u8)
    cv2.imwrite(os.path.join(out, "pseudo_mask_after_noise.png"), noisy_u8)
    cv2.imwrite(os.path.join(out, "pseudo_after_refine.png"), refined_u8)
    cv2.imwrite(os.path.join(out, "pseudo_mask_init.png"), pseudo_u8)
    cv2.imwrite(os.path.join(out, "pseudo_mask_noisy.png"), noisy_u8)
    cv2.imwrite(os.path.join(out, "pseudo_mask_refined.png"), refined_u8)
    x0_u8 = pseudo_u8
    cv2.imwrite(os.path.join(out, "pseudo_mask_x0.png"), x0_u8)
    cv2.imwrite(os.path.join(out, "weak_aug.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, "strong_aug.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, "diffusion_cond.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    n_tiles = len(iter_tile_coords(img_p.shape[0], img_p.shape[1], patch_size, stride))
    print("Resolved IMAGE_PATH:", image_path)
    print("Resolved GT mask:", mask_path_resolved or "(none)")
    print("Original size (H,W):", (h0, w0), "| Padded for sliding:", img_p.shape[:2])
    print("Sliding: patch", patch_size, "stride", stride, "| tiles:", n_tiles)
    print("Outputs: full resolution", (h0, w0), "(no 256×256 exports)")
    print("t_start:", t_start, "of T:", T)
    print("Saved to:", os.path.abspath(out))


if __name__ == "__main__":
    main()
